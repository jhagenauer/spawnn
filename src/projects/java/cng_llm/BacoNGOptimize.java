package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class BacoNGOptimize {

	private static Logger log = Logger.getLogger(BacoNGOptimize.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");

		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/bacao.csv"), new int[]{ 0, 1 }, new int[]{}, true );
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : samples)
			desired.add(new double[] { d[3] });

		final int[] ta = new int[] { 4 };
		final int[] fa = new int[] { 2, 3 };
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0); // should not be necessary

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		final Map<double[],Map<double[],Double>> rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 1);
		GeoUtils.rowNormalizeMatrix(rMap);
		
		// ------------------------------------------------------------------------

		final Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[5];
			if( !ref.containsKey(c) )
				ref.put(c, new HashSet<double[]>());
			ref.get(c).add(d);
		}

		for( final int T_MAX : new int[]{ 10000, 20000, 40000, 60000 } )
			for( final int nrNeurons : new int[]{ 32 } )
			for( final double lInit : new double[]{ nrNeurons, nrNeurons/2, nrNeurons/3 })
			for( final double lFinal : new double[]{ 1.0, 0.1 })	
			for( final double lr1Init : new double[]{ 1.0, 0.5 })
			for( final double lr1Final : new double[]{ 0.005, 0.001 })
			for( final double lr2Init : new double[]{ 0.5, 0.1 })
			for( final double lr2Final : new double[]{ 0.005, 0.001 })
			for (int l = 1; l <= 4; l++ ) {
				final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 4; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						Sorter<double[]> sorter = new KangasSorter<>(new DefaultSorter<>(gDist), new DefaultSorter<>(fDist), L);
						
						DecayFunction nbRate = new PowerDecay(lInit, lFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);
						
						Map<double[],Double> residuals = new HashMap<double[],Double>();
						for( int i = 0; i < samples.size(); i++ )
							residuals.put( samples.get(i),ng.present( samples.get(i) )[0] - desired.get(i)[0] );
						
						// rmse
						double mse = 0;
						for( double d : residuals.values() )
							mse += Math.pow(d, 2);
						double rmse = Math.sqrt( mse/residuals.size() );
						
						// moran
						double[] moran = GeoUtils.getMoransIStatistics(rMap, residuals);
						
						Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
						for (double[] x : samples) {
							sorter.sort(x, ng.getNeurons());
							List<double[]> bmuList = ng.getNeurons();

							Connection c = new Connection(bmuList.get(0), bmuList.get(1));
							if (!conns.containsKey(c))
								conns.put(c, 1);
							else
								conns.put(c, conns.get(c) + 1);
						}
												
						/*for( int i : fa ) {
							Map<double[],Double> values = new HashMap<double[],Double>();
							for( double[] d : neurons )
								values.put(d, d[i]);
							NGUtils.geoDrawNG("output/ng_"+df.format(L)+"_comp_"+sdf.names.get(i)+".png", values, conns.keySet(), ga, samples);
						}
						
						for( int i = 0; i < ng.matrix.values().iterator().next()[0].length; i++ ) {
							Map<double[],Double> values = new HashMap<double[],Double>();
							for( double[] d : neurons )
								values.put(d, ng.matrix.get(d)[0][i]);
							NGUtils.geoDrawNG("output/ng_"+df.format(L)+"_coef_"+sdf.names.get(i+2)+".png", values, conns.keySet(), ga, samples);
						}*/
						
						// build cm of coefficients
						Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
						for( Connection c : conns.keySet() ) {
							double[] a = ng.matrix.get(c.getA())[0];
							double[] b = ng.matrix.get(c.getB())[0];
							if( !cm.containsKey(a) )
								cm.put(a, new HashSet<double[]>() );
							if( !cm.containsKey(b) )
								cm.put(b, new HashSet<double[]>() );
							cm.get(a).add(b);
							cm.get(b).add(a);
						}
						
						Map<Set<double[]>,TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(), Clustering.HierarchicalClusteringType.ward);
						List<Set<double[]>> c = Clustering.cutTree(tree, 3);
						
						List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
						for( Set<double[]> s : c ) {
							Set<double[]> ns = new HashSet<double[]>();
							for( double[] d : s )
								for( double[] n : neurons )
									if( ng.matrix.get(n)[0] == d )
										ns.addAll( mapping.get(n));
							cluster.add(ns);
						}
						double nmi = ClusterValidation.getNormalizedMutualInformation(cluster, ref.values());
																													
						return new double[] {
								rmse,
								moran[0],
								moran[4],
								nmi
						};
					}
				}));
			}
			es.shutdown();
			
			DescriptiveStatistics ds[] = null;
			for (Future<double[]> ff : futures) {
				try {
					double[] ee = ff.get();
					if (ds == null) {
						ds = new DescriptiveStatistics[ee.length];
						for (int i = 0; i < ee.length; i++)
							ds[i] = new DescriptiveStatistics();
					}
					for (int i = 0; i < ee.length; i++)
						ds[i].addValue(ee[i]);
				} catch (InterruptedException ex) {
					ex.printStackTrace();
				} catch (ExecutionException ex) {
					ex.printStackTrace();
				}
			}
						
			try {
				String fn = "output/resultBacao.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,moran,pValue,nmi\n").getBytes());
				}
				String s = T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
