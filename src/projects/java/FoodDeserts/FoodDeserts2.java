package FoodDeserts;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Point;

public class FoodDeserts2 {
	
	private static Logger log = Logger.getLogger(FoodDeserts2.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("/home/julian/tmp/results/results.shp"), true);

		final int[] fa = new int[] { 7, 6, 5, 4, 3 };
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);

		for (int i = 0; i < sdf.samples.size(); i++) {
			double[] d = sdf.samples.get(i);
			Point p = sdf.geoms.get(i).getCentroid();
			d[0] = p.getX();
			d[1] = p.getY();
		}

		DataUtils.transform(sdf.samples, fa, transform.zScore);
		DataUtils.zScoreGeoColumns(sdf.samples, ga, gDist);
		
		String fn = "output/food.csv";
		fn = fn.replaceAll(" ","");
		try {
			Files.write(Paths.get(fn), ("finalNB,initLR,finalLR,fqe,sqe\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}

		final int nrCluster = 9;
		final int nrNeurons = nrCluster;
		final int t_max = 100000;
			
		for (final double finalNB : new double[]{ 0.1, 0.05, 0.01 } ) 	
		for( final double initLR : new double[]{ 0.8, 0.6, 0.4, 0.2, 0.1 } ) 
		for( final double finalLR : new double[]{ 0.01, 0.005, 0.001 } ){	
					
			final int L = 3;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 512; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);

						DecayFunction nbRate = new PowerDecay(nrNeurons * 2.0 / 3.0, finalNB);
						DecayFunction lrRate1 = new PowerDecay(initLR, finalLR);

						List<double[]> neurons = new ArrayList<double[]>();
						while (neurons.size() < nrNeurons) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						NG ng = new NG(neurons, nbRate, lrRate1, sorter);
						for (int t = 0; t < t_max; t++) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							ng.train((double) t / t_max, d);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(sdf.samples, neurons, sorter);
						
						if( nrNeurons > nrCluster ) {
														
							Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
							for( double[] x : sdf.samples ) {
								sorter.sort(x,  neurons );
								double[] a = neurons.get(0);
								double[] b = neurons.get(1);
								
								if( !cm.containsKey(a) )
									cm.put(a, new HashSet<double[]>());
								if( !cm.containsKey(b) )
									cm.put(b, new HashSet<double[]>());
								cm.get(a).add(b);
								cm.get(b).add(a);
							}
													
							Map<Set<double[]>, TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
							
							List<Set<double[]>> c = new ArrayList<Set<double[]>>();
							for (Set<double[]> s : Clustering.cutTree(tree, nrCluster )) {
								Set<double[]> set = new HashSet<double[]>();
								for (double[] p : s)
									set.addAll(bmus.get(p));
								c.add(set);
							}
							return new double[] { DataUtils.getWithinClusterSumOfSuqares(c, fDist), DataUtils.getWithinClusterSumOfSuqares(c, gDist) };
						} else {
							return new double[] { DataUtils.getWithinClusterSumOfSuqares(bmus.values(), fDist), DataUtils.getWithinClusterSumOfSuqares(bmus.values(), gDist) };
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
			try {
				Files.write(Paths.get(fn), (finalNB+","+initLR+","+finalLR+","+ds[0].getMean()+","+ds[1].getMean()+"\n").getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
