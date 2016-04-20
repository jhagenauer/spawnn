package eval_geosom;

import java.util.ArrayList;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class EvalGeoSOM_EdgeEffect {

	private static Logger log = Logger.getLogger(EvalGeoSOM_EdgeEffect.class);

	public static void main(String[] args) {
		final Random r = new Random();

		int threads = 4;
		final int MAX = 32;
		final int X_DIM = 3;
		final int T_MAX = 100000;

		try {
			
			for (int k : new int[]{0} ) {
				log.debug("k: "+k);

				final List<double[]> samples = new ArrayList<double[]>();
				for( int i = 0; i < T_MAX; i++ ) {
					double x = r.nextDouble();
					double[] d = new double[]{x, Math.pow(x, 2.0)}; 
					samples.add(d);
				}
				
				int[] ga = new int[] { 0 };
				int[] fa = new int[] { 1 };

				final Dist<double[]> gDist = new EuclideanDist(ga);
				final Dist<double[]> fDist = new EuclideanDist(fa);

				for (int method : new int[] { 0, 1 }) {
					if (method == 0) { // geosom

						for (int radius : new int[]{ 1 } ) {
							final int RADIUS = radius;
							
							ExecutorService es = Executors.newFixedThreadPool(threads);
							List<Future<double[][]>> futures = new ArrayList<Future<double[][]>>();

							for (int i = 0; i < MAX; i++) {

								futures.add(es.submit(new Callable<double[][]>() {

									@Override
									public double[][] call() throws Exception {
										Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, 1);
												//new Grid2DToroid<double[]>(3, 1);
										SomUtils.initRandom(grid, samples);

										KangasBmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, RADIUS);
										SOM som = new SOM(new GaussKernel(new LinearDecay(X_DIM, 0.1)), new LinearDecay(1.0, 0.005), grid, bg);
										for (int t = 0; t < T_MAX; t++) {
											double[] x = samples.get(r.nextInt(samples.size()));
											som.train((double) t / T_MAX, x);
										}
										
										Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bg);
										List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
										Collections.sort(pos);
										
										double[][] r = new double[2][grid.size()];
										for( int i = 0; i < pos.size(); i++ ) {
											GridPos p = pos.get(i);
											double[] w = grid.getPrototypeAt(p);
											r[0][i] = DataUtils.getQuantizationError(w, mapping.get(p), fDist);
											r[1][i] = DataUtils.getQuantizationError(w, mapping.get(p), gDist);
										}
										return r;
									}
								}));
							}
							es.shutdown();
							
							// RMSE
							double fError = 0, sError = 0;
							for( Future<double[][]> f : futures ) {
								double[][] d = f.get();
								fError += Math.pow(d[0][0] - d[0][1], 2);
								sError += Math.pow(d[1][0] - d[1][1], 2);
								
							}
							fError = Math.sqrt( fError/futures.size() );
							sError = Math.sqrt( sError/futures.size() );
							
							log.debug("geosom: "+RADIUS);
							log.debug("Error fDist: "+fError);
							log.debug("Errpr gDist: "+sError);
						}
									
					} else { // wsom
						
						for (int w : new int[]{0,10,20,30,40,50,60,70,80,90,100}) {
							final double W = (double) w / 100;
							
							ExecutorService es = Executors.newFixedThreadPool(threads);
							List<Future<double[][]>> futures = new ArrayList<Future<double[][]>>();

							for (int i = 0; i < MAX; i++) {

								futures.add(es.submit(new Callable<double[][]>() {

									@Override
									public double[][] call() throws Exception {

										Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, 1);
												//new Grid2DToroid<double[]>(3, 1);
										SomUtils.initRandom(grid, samples);

										Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
										m.put(fDist, W);
										m.put(gDist, 1.0 - W);
										BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(new WeightedDist<double[]>(m));

										SOM som = new SOM(new GaussKernel(new LinearDecay(X_DIM, 0.1)), new LinearDecay(1.0, 0.005), grid, bg);
										for (int t = 0; t < T_MAX; t++) {
											double[] x = samples.get(r.nextInt(samples.size()));
											som.train((double) t / T_MAX, x);
										}
										
										Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bg);
										List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
										Collections.sort(pos);
										
										double[][] r = new double[2][grid.size()];
										for( int i = 0; i < pos.size(); i++ ) {
											GridPos p = pos.get(i);
											double[] w = grid.getPrototypeAt(p);
											r[0][i] = DataUtils.getQuantizationError(w, mapping.get(p), fDist);
											r[1][i] = DataUtils.getQuantizationError(w, mapping.get(p), gDist);
										}
										return r;
									}
								}));
							}
							es.shutdown();

							// RMSE
							double fError = 0, sError = 0;
							for( Future<double[][]> f : futures ) {
								double[][] d = f.get();
								fError += Math.pow(d[0][0] - d[0][1], 2);
								sError += Math.pow(d[1][0] - d[1][1], 2);
								
							}
							fError /= futures.size();
							sError /= futures.size();
							
							log.debug("wsom: "+W);
							log.debug("Error fDist: "+fError);
							log.debug("Errpr gDist: "+sError);
							
						}
					}
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
	}
}
