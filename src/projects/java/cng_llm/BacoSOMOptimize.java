package cng_llm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import llm.LLMSOM;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class BacoSOMOptimize {

	private static Logger log = Logger.getLogger(BacoSOMOptimize.class);

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

		for( final int T_MAX : new int[]{ 40000 } )
			for( final double nbInit : new double[]{ 8 })
			for( final double nbFinal : new double[]{ 0.1 })	
			for( final double lr1Init : new double[]{ 0.5 })
			for( final double lr1Final : new double[]{ 0.001 })
			for( final double lr2Init : new double[]{ 0.1 })
			for( final double lr2Final : new double[]{ 0.001 })
			for (int l = 1; l <= 13; l++ ) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(1);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 4; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						Grid2DHex<double[]> grid = new Grid2DHex<>(12, 8);
						SomUtils.initRandom(grid, samples);
						
						KernelFunction nb1 = new GaussKernel( new LinearDecay(nbInit, nbFinal));
						KernelFunction nb2 = new GaussKernel( new LinearDecay(nbInit, nbFinal));
						DecayFunction lr1 = new LinearDecay(lr1Init, lr1Final);
						DecayFunction lr2 = new LinearDecay(lr2Init, lr2Init);
						BmuGetter<double[]> bmuGetter = new KangasBmuGetter<>(new DefaultBmuGetter<>(gDist), new DefaultBmuGetter<>(fDist), L);

						LLMSOM som = new LLMSOM( 
								nb1, lr1, 
								grid, bmuGetter, 
								nb2, lr2, 
								fa, 1);
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							som.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
						
						Map<double[],Double> residuals = new HashMap<double[],Double>();
						for( int i = 0; i < samples.size(); i++ )
							residuals.put( samples.get(i),som.present( samples.get(i) )[0] - desired.get(i)[0] );
						
						// rmse
						double mse = 0;
						for( double d : residuals.values() )
							mse += Math.pow(d, 2);
						double rmse = Math.sqrt( mse/residuals.size() );
						
						// moran
						double[] moran = GeoUtils.getMoransIStatistics(rMap, residuals);
						
						// cluster coefficients and get nmi
						Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
						for( GridPos p : grid.getPositions() ) {
							Set<double[]> s = new HashSet<double[]>();
							for( GridPos nb : grid.getNeighbours(p) )
								s.add( som.matrix.get(nb)[0] );
							cm.put( som.matrix.get(p)[0], s );
						}
						
						List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(), Clustering.HierarchicalClusteringType.ward);
						List<Set<double[]>> c = Clustering.cutTree(tree, 3);
						
						List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
						for( Set<double[]> s : c ) {
							Set<double[]> ns = new HashSet<double[]>();
							for( double[] d : s )
								for( GridPos p : grid.getPositions() )
									if( som.matrix.get(p)[0] == d )
										ns.addAll( mapping.get(p));
							cluster.add(ns);
						}
						double nmi = ClusterValidation.getNormalizedMutualInformation(cluster, ref.values());
						
						for( int i : fa ) 
							SomUtils.printComponentPlane(grid, i, ColorBrewer.Blues, new FileOutputStream("output/grid_"+df.format(L)+"_comp_"+sdf.names.get(i)+".png"));
						
						Grid2DHex<double[]> gridCoefs = new Grid2DHex<>(12,8);
						for( GridPos p : grid.getPositions() ) 
							gridCoefs.setPrototypeAt(p, som.matrix.get(p)[0]);
						
						for( int i = 0; i < gridCoefs.getPrototypes().iterator().next().length; i++ ) 
							SomUtils.printComponentPlane(gridCoefs, i, ColorBrewer.Reds, new FileOutputStream("output/grid_"+df.format(L)+"_coef_"+i+".png"));
																													
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
					Files.write(Paths.get(fn), ("t_max,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,moran,pValue,nmi\n").getBytes());
				}
				String s = T_MAX+","+l+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
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
