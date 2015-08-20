package growing_cng;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class GrowingCNG_Housing_Thread {

	private static Logger log = Logger.getLogger(GrowingCNG_Housing_Thread.class);

	/* Note: Submarket-Stuff does not totally fit in here: Needed GWR + Kriging
	 * 
	 */
	public static void main(String[] args) {
		final Random r = new Random();
	
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
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
		}*/
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng/test2c.shp"), true);
		List<String> vars = new ArrayList<String>();
		vars.add("X");
		vars.add("Y");
		vars.add("VALUE");
		
		for (double[] d : sdf.samples) {
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			geoms.add(sdf.geoms.get(idx));
			desired.add( new double[]{ r.nextDouble() });
		}
				
		int[] fa = new int[vars.size() - 2];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };
		DataUtils.zScoreColumns(samples, fa);
		
		/*
		// PCA
		int nrComponents = 1;
		List<double[]> ns = DataUtils.removeColumns(samples, ga);
		ns = DataUtils.reduceDimensionByPCA(ns, nrComponents, true);
		for (int k = 0; k < ns.size(); k++) {
			double[] nd = new double[ga.length + nrComponents];
			for (int i = 0; i < ga.length; i++)
				nd[i] = samples.get(k)[ga[i]];
			for (int i = 0; i < nrComponents; i++)
				nd[i + ga.length] = ns.get(k)[i];
			samples.set(k, nd);
		}
		final int[] nFa = new int[nrComponents];
		for (int i = 0; i < nrComponents; i++)
			nFa[i] = i + ga.length;*/
										
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		boolean firstWrite = true;

		/* Was kann man machen, um Einfluß von fistMode zu maximieren?
		 * - Kleine Adaptionsrate (verlängert weg von sclhechten neuronen zu guten positionen)
		 * - Großes aMax (viele Neuronen machen die richtige Wahl für Insert kritischer?!)
		 */
			
		for( final int T_MAX : new int[]{ 40000 } )
		for (final double lrB : new double[] { 0.05 }) 
			for (final double lrN : new double[] { lrB/100 })
				for (final int lambda : new int[] { 600, 900, 1200 }) // 300 bringts gar nichts, besser 600 evtl. auch 900
					for (final int aMax : new int[] { lambda/3 })
						for( final double alpha : new double[]{ 0.45 } ) //  0.45 oder 0.5, kein großer unterschied
							for( final double beta : new double[]{ 0.00005 } )
								for( double ratio : new double[]{ 1.0 } )
								//for( double ratio = 0.0; (ratio+=0.01) <= 1.0; )
								for( final int distMode : new int[]{ -1, 0, 2, 7, 8 } ){
																														

							final double RATIO = ratio;
							ExecutorService es = Executors.newFixedThreadPool(4);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
							
							for (int i = 0; i < 16; i++) {
								final int RUN = i;
								futures.add(es.submit(new Callable<double[]>() {

									@Override
									public double[] call() throws Exception {

										List<double[]> neurons = new ArrayList<double[]>();
										for (int i = 0; i < 2; i++) {
											double[] d = samples.get(r.nextInt(samples.size()));
											neurons.add(Arrays.copyOf(d, d.length));
										}

										GrowingCNG ng = new GrowingCNG(neurons, lrB, lrN, gDist, fDist, RATIO, aMax, lambda, alpha, beta );
										ng.samples = samples;
										ng.distMode = distMode;
										ng.run = RUN;
										
										int rate = 100;
										List<Double> fQes = new ArrayList<Double>();
										List<Double> sQes = new ArrayList<Double>();
										int t = 1;
										while( true ) { 
											double[] x = samples.get(r.nextInt(samples.size())); 
											ng.train(t, x);
																																									
											if (t % rate == 0) {
												Map<double[],Set<double[]>> mapping = ng.getMapping(samples);
												fQes.add(DataUtils.getMeanQuantizationError(mapping, fDist));
												sQes.add(DataUtils.getMeanQuantizationError(mapping, gDist));
											}
																						
											if( t >= T_MAX ) // normal time break
												break;
											t++;
										}
										
										Map<double[], Set<double[]>> mapping = ng.getMapping(samples); 
										List<double[]> usedNeurons = new ArrayList<double[]>(mapping.keySet());
										
										List<double[]> response = new ArrayList<double[]>();
										List<Integer> toIgnore = new ArrayList<Integer>();
										for (int i : ga)
											toIgnore.add(i);
										double[] y = new double[samples.size()];
										double[][] x = new double[samples.size()][];
										int l = 0;
										for (double[] d : samples) {
											int idx = samples.indexOf(d);
											y[l] = desired.get(idx)[0];
											x[l] = new double[d.length - toIgnore.size() + usedNeurons.size() - 1];
											int j = 0;
											for (int i = 0; i < d.length; i++) {
												if (toIgnore.contains(i))
													continue;
												x[l][j++] = d[i];
											}
											// add dummy variable
											for (int k = 1; k < usedNeurons.size(); k++)
												if (mapping.get(usedNeurons.get(k)).contains(d))
													x[l][j + k - 1] = 1;											
											l++;
										}

										OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
										ols.setNoIntercept(false);
										ols.newSampleData(y, x);
										double[] beta = ols.estimateRegressionParameters();

										for (double[] d : samples) {
											double[] nd = new double[d.length - toIgnore.size() + usedNeurons.size() - 1];
											int j = 0;
											for (int i = 0; i < d.length; i++) {
												if (toIgnore.contains(i))
													continue;
												nd[j++] = d[i];
											}
											// add dummy variable
											for (int k = 1; k < usedNeurons.size(); k++)
												if (mapping.get(usedNeurons.get(k)).contains(d))
													nd[j + k - 1] = 1;

											double ps = beta[0]; // intercept at beta[0]
											for (int k = 1; k < beta.length; k++)
												ps += beta[k] * nd[k - 1];

											response.add(new double[] { ps });
										}
										double mse = Meuse.getMSE(response, desired); // variance of residuals
										double aic = samples.size() * Math.log( mse ) + 2 * (beta.length+1);
										

										Double minFQe = Collections.min(fQes);
										DescriptiveStatistics ds10FQe = new DescriptiveStatistics();
										for( double d : fQes.subList(fQes.size()-10, fQes.size() ) )
											ds10FQe.addValue(d);
										
										Double minSQe = Collections.min(sQes);
										DescriptiveStatistics ds10SQe = new DescriptiveStatistics();
										for( double d : sQes.subList(sQes.size()-10, sQes.size() ) )
											ds10SQe.addValue(d);
										
										double[] r = new double[] {
												t,
												ng.getNeurons().size(), 
												usedNeurons.size(), 
												ng.getConections().size(), 
												
												minSQe,
												sQes.indexOf(minSQe)*rate,
												DataUtils.getMeanQuantizationError(mapping, gDist),
												ds10SQe.getMean(),
												ds10SQe.getStandardDeviation(),
												
												minFQe,
												fQes.indexOf(minFQe)*rate,
												DataUtils.getMeanQuantizationError(mapping, fDist), 	
												ds10FQe.getMean(),
												ds10FQe.getStandardDeviation(),
												
												Math.sqrt(mse),
												aic,
												ng.aErrorSum,
												ng.bErrorSum
												};
																					
										return r;
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
								String fn = "output/result.csv";
								if( firstWrite ) {
									firstWrite = false;
									Files.write(Paths.get(fn), "alpha,beta,lambda,mode,ratio,t,neurons,used,connections,minSqe,minSqeTime,finalSqe,mean10Sqe,sd10Sqe,minFqe,minFQeTime,finalFqe,mean10Fqe,sd10Fqe,rmse,aic,aErrorSum,bErrorSum\n".getBytes());
									/*for( int i = 0; i < ds.length; i++ )
										fw.write(",p_"+i);
									fw.write("\n");*/
								}
								String s = alpha+","+beta+","+lambda+","+"mode_"+distMode+","+ratio;
								for (int i = 0; i < ds.length; i++)
									s += ","+ds[i].getMean();
								s += "\n";
								Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
								System.out.println(s.substring(0, Math.min(s.length(),256)));
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
	}
}
