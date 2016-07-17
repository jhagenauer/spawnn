package foodDeserts;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
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
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

public class FoodDeserts {
	
	private static Logger log = Logger.getLogger(FoodDeserts.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("/home/julian/tmp/results/results.shp"), true);

		final int[] fa = new int[] { 4,5,18,19,23 };
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);

		for (int i = 0; i < sdf.samples.size(); i++) {
			double[] d = sdf.samples.get(i);
			Point p = sdf.geoms.get(i).getCentroid();
			d[0] = p.getX();
			d[1] = p.getY();
		}

		DataUtils.transform(sdf.samples, fa, Transform.zScore);
		DataUtils.zScoreGeoColumns(sdf.samples, ga, gDist);
		
		String fn = "output/food.csv";
		fn = fn.replaceAll(" ","");
		try {
			Files.write(Paths.get(fn), ("neurons,radius,error,value\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}

		final int nrCluster = 9;
		final int t_max = 100000;
		final int maxRun = 1;
		for( final int nrNeurons : new int[]{ nrCluster/*, nrCluster*3,nrCluster*6*/ } )
		for (int l : new int[]{2,3}) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < maxRun; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);

						DecayFunction nbRate = new PowerDecay(nrNeurons * 2.0 / 3.0, 0.1);
						DecayFunction lrRate1 = new PowerDecay(0.2, 0.005);

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
						
						if( maxRun == 1 ) {
							List<double[]> nSamples = new ArrayList<double[]>();
							for( double[] d : sdf.samples ) {
								double[] nd = Arrays.copyOf(d, d.length+1);
								for( double[] n : neurons )
									if( bmus.get(n).contains(d) )
										nd[nd.length-1] = neurons.indexOf(n)+1;
								nSamples.add(nd);
							}
							List<String> nNames = new ArrayList<String>(sdf.names);
							nNames.add("cngCluster");
							
							DataUtils.writeShape(nSamples, sdf.geoms, nNames.toArray(new String[]{} ), sdf.crs, "output/cng_results_"+L+".shp");
							Drawer.geoDrawValues(sdf.geoms, nSamples, sdf.samples.get(0).length, sdf.crs, ColorBrewer.Set3, "output/cng_cluster_"+L+".png");
						}
						
						return new double[] { DataUtils.getWithinSumOfSquares(bmus.values(), fDist), DataUtils.getWithinSumOfSquares(bmus.values(), gDist) };
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
				Files.write(Paths.get(fn), (nrNeurons+","+L+",fqe,"+ds[0].getMean()+"\n").getBytes(), StandardOpenOption.APPEND);
				Files.write(Paths.get(fn), (nrNeurons+","+L+",sqe,"+ds[1].getMean()+"\n").getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
