package foodDeserts;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

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
import spawnn.utils.DataUtils.transform;
import spawnn.utils.GraphClustering;
import spawnn.utils.SpatialDataFrame;

public class FoodDeserts3 {

	private static Logger log = Logger.getLogger(FoodDeserts3.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/foodDeserts2/data_sel.shp"), true);
				
		final int[] fa = new int[] { 129, 128, 127, 136, 133, 134 };
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

		String fn = "output/food3.csv";
		try {
			Files.write(Paths.get(fn), ("").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		final int t_max = 100000;
		
		ExecutorService es = Executors.newFixedThreadPool(4);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		for( int run = 0; run < 10; run++ )
		for (int n : new int[]{72} )
			for (int l : new int[]{ 34 }) {
				
				final int[] pars = new int[]{run,n,l};

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, pars[2]);

						DecayFunction nbRate = new PowerDecay(pars[1] * 2.0 / 3.0, 0.01);
						DecayFunction lrRate1 = new PowerDecay(0.6, 0.005);

						List<double[]> neurons = new ArrayList<double[]>();
						while (neurons.size() < pars[1]) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						NG ng = new NG(neurons, nbRate, lrRate1, sorter);
						for (int t = 0; t < t_max; t++) {
							double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
							ng.train((double) t / t_max, d);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(sdf.samples, neurons, sorter);
						double qe = DataUtils.getMeanQuantizationError(bmus, fDist);
						double sqe = DataUtils.getMeanQuantizationError(bmus, gDist);
						
						Map<double[],Map<double[],Double>> graph = new HashMap<>();
						for( double[] d : sdf.samples ) {
							sorter.sort(d, neurons);					
							double[] n0 = neurons.get(0), n1 = neurons.get(1);
							
							if( !graph.containsKey( n0 ) )
								graph.put(n0, new HashMap<double[],Double>() );
							if( !graph.get(n0).containsKey(n1) )
								graph.get(n0).put(n1, 1.0);
							else
								graph.get(n0).put(n1, graph.get(n0).get(n1)+1.0);
							
							if( !graph.containsKey( n1) )
								graph.put(n1, new HashMap<double[],Double>() );
							if( !graph.get(n1).containsKey(n0) )
								graph.get(n1).put(n0, 1.0);
							else
								graph.get(n1).put(n0, graph.get(n1).get(n0)+1.0);
						}
						
						double max = 0;
						for( Map<double[],Double> m : graph.values() )
							max = Math.max( max, Collections.max(m.values() ) );
						
						Map<double[],Map<double[],Double>> nGraph = new HashMap<>();
						for( double[] n0 : graph.keySet() ) {
							Map<double[],Double> m = new HashMap<>();
							for( double[] n1 : graph.get(n0).keySet() )
								m.put(n1, graph.get(n0).get(n1)/max );
							nGraph.put(n0, m);
						}
												
						Map<double[],Integer> map = GraphClustering.multilevelOptimization(nGraph, 100 );
						
						Map<double[],Set<double[]>> cluster = new HashMap<>();
						for( Set<double[]> c :GraphClustering.modulMapToCluster(map) ) 
							cluster.put(DataUtils.getMean(c), c);
						
						double sc = ClusterValidation.getSilhouetteCoefficient(cluster, fDist);
						double mod = GraphClustering.modularity(nGraph, map);
						return new double[]{pars[0],pars[1],pars[2],sc,mod};
					}
				}));
			}
		es.shutdown();

		for (Future<double[]> ff : futures) {
			try {
				double[] e = ff.get();
				String s = (Arrays.toString(e).replace("[", "").replace("]", "") + "\n");
				System.out.print(s);
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}		
	}
}
