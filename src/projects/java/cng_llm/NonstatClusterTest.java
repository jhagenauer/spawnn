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

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class NonstatClusterTest {

	private static Logger log = Logger.getLogger(NonstatClusterTest.class);
	
	enum method { error, coef, inter, y };

	public static void main(String[] args) {
		final Random r = new Random();
		DecimalFormat df = new DecimalFormat("00");

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng_llm/nonstatCluster.shp"), true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		final int ta = 3; // y-index
		final int[] fa = new int[] { 2 }; // x-index
		final int[] ga = new int[] { 0, 1 };
		
		for (double[] d : samples)
			desired.add(new double[] { d[ta] });

		// DataUtils.zScoreColumns(samples, fa);
		// DataUtils.zScoreColumn(samples, ta); // should not be necessary

		final Map<Integer, Set<double[]>> ref = new HashMap<Integer, Set<double[]>>();
		for (double[] d : samples) {
			int c = (int) d[4];
			if (!ref.containsKey(c))
				ref.put(c, new HashSet<double[]>());
			ref.get(c).add(d);
		}
		
		final List<double[]> coefs = DataUtils.readDataFrameFromCSV(new File("output/x1_coef.csv"), new int[]{}, false).samples;
		final List<double[]> inter = DataUtils.readDataFrameFromCSV(new File("output/intercept.csv"), new int[]{}, false).samples;

		final int T_MAX = 40000;
		final int nrNeurons = 25;

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> tDist = new EuclideanDist(new int[]{ta});

		final DecayFunction nbRate = new PowerDecay(nrNeurons / 3, 0.1);
		final DecayFunction lrRate1 = new PowerDecay(0.5, 0.005);
		final DecayFunction lrRate2 = new PowerDecay(0.1, 0.005);
		
		String fn = "output/nonstatClusterResult.csv";
		try {
			Files.write(Paths.get(fn), ("method,l,nmi\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for( final method m : method.values() ) {
		log.debug(m);
		for (int l = 1; l <= 25; l++) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 32; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}
						
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						if( m == method.error ) {
							Sorter<double[]> secSorter = new ErrorSorter(samples, desired);
							Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
							LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1, false);
							((ErrorSorter)secSorter).setLLMNG(ng);
							
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samples.size());
								ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
							}
							Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
							return new double[]{ClusterValidation.getNormalizedMutualInformation(mapping.values(), ref.values())};	
							
						} else if( m == method.y ) {
							Sorter<double[]> secSorter = new DefaultSorter<>( tDist );
							Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
							NG ng = new NG(neurons, nbRate, lrRate1, sorter );
							
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samples.size());
								ng.train( (double) t / T_MAX, samples.get(j) );
							}
							Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
							return new double[]{ClusterValidation.getNormalizedMutualInformation(mapping.values(), ref.values())};	
						} else {
							List<Double> old = new ArrayList<Double>();
							for( int i = 0; i < samples.size(); i++ ) {
								double[] d = samples.get(i);
								old.add(d[fa[0]]);
								if( m == method.coef )
									d[fa[0]] = coefs.get(i)[0];
								else if( m == method.inter )
									d[fa[0]] = inter.get(i)[0];
							}
							
							Sorter<double[]> secSorter = new DefaultSorter<>( fDist );
							Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
							NG ng = new NG(neurons, nbRate, lrRate1, sorter );
							
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samples.size());
								ng.train( (double) t / T_MAX, samples.get(j) );
							}
							Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
							
							for( int i = 0; i < samples.size(); i++ ) { // restore
								double[] d = samples.get(i);
								d[fa[0]] = old.get(i);
							}
							
							return new double[]{ClusterValidation.getNormalizedMutualInformation(mapping.values(), ref.values())};	
						}
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
			
			String s = m+","+L;
			for (int i = 0; i < ds.length; i++)
				s += ","+ds[i].getMean();
			s += "\n";
			try {
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e1) {
				e1.printStackTrace();
			}

		}
		}

	}
}
