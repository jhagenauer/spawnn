package llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class OptimizeHousingSOM {

	private static Logger log = Logger.getLogger(OptimizeHousingSOM.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();

		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			desired.add(new double[] { d[sdf.names.indexOf("lnp")] });
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samples, fa);
		
		int samplesSize = samples.size();
		final List<double[]> samplesVal = new ArrayList<double[]>();
		final List<double[]> desiredVal = new ArrayList<double[]>();
		while( samplesVal.size() < 0.3*samplesSize) {
			int idx = r.nextInt(samples.size());
			samplesVal.add(samples.remove(idx));
			desiredVal.add(desired.remove(idx));
		}

		final Dist<double[]> gDist = new EuclideanDist(new int[] { 0, 1 });
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		double[][] bestParams = new double[3][];
		double bestMean[] = new double[bestParams.length];
		for (final double a : new double[] { 7 })
			for (final double b : new double[] { 0.1 })
				for (final double c : new double[] { 1.0 })
					for (final double d : new double[] { 0.005 })
						for (final double aa : new double[] { 7, 5, 3, 2 })
							for (final double bb : new double[] { 0.1, 0.01 })
								for (final double cc : new double[] { 0.75, 0.5, 0.25 })
									for (final double dd : new double[] { 0.01, 0.005, 0.001 })
										for( final double e : new double[] { 0, 1 } )
											for( final double f : new double[] { 0, 1 } )
												for( final double g : new double[] { 0, 1 } )
													for( final double h : new double[] { 0, 1 } )
															{
						if (a <= b || c <= d)
							continue;

						ExecutorService es = Executors.newFixedThreadPool(4);
						List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

						for (int i = 0; i < 12; i++) {
							futures.add(es.submit(new Callable<double[]>() {

								@Override
								public double[] call() throws Exception {
									/*BmuGetter<double[]> bmuGetterA = new DefaultBmuGetter<double[]>(fDist);
									BmuGetter<double[]> bmuGetterB = new ErrorBmuGetter(samples, desired);
									BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(bmuGetterA, bmuGetterB, (int)k);*/
									BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);
																		
									Grid2D<double[]> grid = new Grid2DHex<double[]>(6,5);
									SomUtils.initRandom(grid, samples);
									
									DecayFunction da = new LinearDecay(a, b);
									DecayFunction db = new LinearDecay(c, d);
									DecayFunction daa = new LinearDecay(aa, bb);
									DecayFunction dbb = new LinearDecay(cc, dd);
									
									if( e == 1 )
										da = new PowerDecay(a, b);
									if( f == 1 )
										db = new PowerDecay(c, d);
									if( g == 1 )
										daa = new PowerDecay(aa, bb);
									if( h == 1 )
										dbb = new PowerDecay(cc, dd);
									
																																													
									LLMSOM som = new LLMSOM(
											new GaussKernel( da ), db, 
											grid, bmuGetter,            
											new GaussKernel( daa ), dbb,
											fa, 1);
										
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(samples.size());
										som.train((double) t / T_MAX, samples.get(j), desired.get(j));
									}
									
									/*bmuGetter = bmuGetterA;
									som.setBmuGetter(bmuGetter);*/
									
									List<double[]> responseVal = new ArrayList<double[]>();
									for (double[] x : samplesVal)
										responseVal.add(som.present(x));
									
									double rmse = Meuse.getRMSE(responseVal, desiredVal);
									if( Double.isNaN(rmse))
										rmse = Double.MAX_VALUE;
									return new double[] { rmse, 
											1.0 - Math.pow(Meuse.getPearson(responseVal, desiredVal),2.0), 
											SomUtils.getQuantizationError(grid, bmuGetter, fDist, samplesVal) };
								}
							}));
						}
						es.shutdown();

						double[] mean = new double[bestMean.length];
						for (Future<double[]> ff : futures) {
							try {
								double[] ee = ff.get();
								for (int i = 0; i < mean.length; i++)
									mean[i] += ee[i] / futures.size();
							} catch (InterruptedException ex) {
								ex.printStackTrace();
							} catch (ExecutionException ex) {
								ex.printStackTrace();
							}
						}

						for (int i = 0; i < mean.length; i++) {
							if (bestParams[i] == null || mean[i] < bestMean[i]) {
								bestParams[i] = new double[] { a, b, c, d, aa, bb, cc, dd, e, f, g, h };
								bestMean[i] = mean[i];
								log.debug(i + ":" + Arrays.toString(bestParams[i]) + "," + Arrays.toString(mean) );
							}
						}

					}

	}
}
