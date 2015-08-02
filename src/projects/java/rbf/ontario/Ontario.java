package rbf.ontario;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.AdaptIncRBF;
import spawnn.rbf.RBF;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class Ontario {

	private static Logger log = Logger.getLogger(Ontario.class);
	
	public static void main(String[] args) {
		final Random r = new Random();
		
		SpatialDataFrame sd = DataUtils.readSpatialDataFrameFromShapefile(new File("data/ontario/clipped/ontario_inorg_sel_final.shp"), true);
		DataUtils.writeCSV("output/ontario.csv", sd.samples, sd.names.toArray(new String[]{}));
		
		Map<String, DescriptiveStatistics[]> results = new HashMap<String, DescriptiveStatistics[]>();
			
		// false (100 runs):
		//0,05	false	5000	0,05	50	50000	14	0,05	4000	0,8171	0,7926408523	0,3866373125	8,331	0,0207825627 // Co
		//0,05	false	5000	0,05	50	50000	64	0,05	4000	0,8636	0,8356449561	0,3226652925	8,905	0,0194709643 // Cr_1
		//0,05	false	5000	0,05	50	50000	33	0,05	4000	0,7905	0,7568949743	0,4419900927	8,085	0,0173149599 // Ni
		//0,05	false	5000	0,05	50	50000	57	0,05	4000	0,9101	0,8964514058	0,2255775117	8,992	0,0147327451 // Al
		
		// true (200 runs, no adapt)
		// 1.5,0.01,true,5000,0.05,70,60000,33,0.0,4000,1.0,0.7549603477666164,0.4562181269616205,6.396,0.0
		// 1.5,0.0050,true,5000,0.05,70,60000,33,0.0,4000,1.0,0.7491505704895403,0.46121275681522356,7.894,0.0
					
		double noAdaptCor = 0;
				
		for( final boolean idw : new boolean[]{ true } )
		for( int t_max : new int[]{ 60000 } )
		for( final int ins_per : new int[]{ 5000 } )
		for( final double delta : new double[]{ 0.05 } )
		for( final int a_max : new int[]{ 70 } )
		for( double exp : new double[]{ 1.5 } )
		for( int n : new int[]{ 14 } ) { // co (14), ni (33), V (51)
		//for( int n = 7; n <= 77; n++ ) {
		List<double[]> all = new ArrayList<double[]>();
		EuclideanDist dist23 = new EuclideanDist(new int[] { 2, 3 });
					
		for (double[] d : sd.samples ) {
			// get IWD
			double iwd = 0;
			for (double[] d2 : sd.samples ) {
				if (d2 == d) 
					continue;
				iwd += d2[n] / Math.pow( dist23.dist(d, d2), exp );
			}
			all.add( new double[]{ d[2], d[3], iwd, d[n]} );
		}
																		
		int[] ga = new int[]{0,1};
		DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga) );
		DataUtils.zScoreColumns(all, new int[]{ 2, 3 } );
			
		for( final double lrA : new double[] { 0.005 }  )
		for( double sc : new double[]{ 0.0, 0.05 } )
		for( final int adapt_per : new int[]{ 500, 1000, 2000, 4000 } )
		for( final int aMode : new int[]{ 0, 1} ) {
			if( sc == 0.0 && ( adapt_per > 500 || aMode > 0 ) )
				continue;
						
			final double SC = sc;
			final int T_MAX = t_max;
																
			final Dist<double[]> dist = new EuclideanDist();
			final int maxK = 10;
							
			ExecutorService es = Executors.newFixedThreadPool( 15 );
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
			for (int j = 0; j < 32; j++) {
				Collections.shuffle(all);
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();

				for( double[] d : all ) {
					if( idw )
						samples.add(new double[] { d[0], d[1], d[2] });
					else
						samples.add(new double[] { d[0], d[1] });
					desired.add(new double[] { d[3] });
				}
				
				for (int k = 0; k < maxK; k++) {
					final int K = k;

					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {
							List<double[]> training = new ArrayList<double[]>();
							List<double[]> trainingDesired = new ArrayList<double[]>();
							List<double[]> validation = new ArrayList<double[]>();
							List<double[]> validationDesired = new ArrayList<double[]>();

							for (int i = 0; i < samples.size(); i++) {
								if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
									validation.add(samples.get(i));
									validationDesired.add(desired.get(i));
								} else {
									training.add(samples.get(i));
									trainingDesired.add(desired.get(i));
								}
							}

							Map<double[], Double> hidden = new HashMap<double[], Double>();
							while (hidden.size() < 2) {
								double[] d = samples.get(r.nextInt(samples.size()));
								hidden.put(Arrays.copyOf(d, d.length), 1.0);
							}

							//AdaptIncRBF rbf = new AdaptIncRBF(hidden, 0.005, 0.0005, dist, a_max, SC, 0.0005, 0.5, delta, 1);
							AdaptIncRBF rbf = new AdaptIncRBF(hidden, lrA, 0.0005, dist, a_max, SC, 0.0005, 0.5, delta, 1);
														
							double[] errors = new double[T_MAX];
							
							for (int t = 0; t < T_MAX; t++) {
								int idx = r.nextInt(training.size());																
								rbf.train(training.get(idx), trainingDesired.get(idx));		
										
								if( aMode == 0 )
									rbf.adaptScale( rbf.getTotalError() ); // conventional
								else 
									rbf.adaptScale( Ontario.getRMSE(rbf, training, trainingDesired) ); 
								
								/*int d = T_MAX / errors.length;
								if (t % d == 0) {
									List<double[]> response = new ArrayList<double[]>();
									for (double[] x : validation)
										response.add(rbf.present(x));
									errors[t / d] = Meuse.getRMSE(response, validationDesired);
								}*/
							}
							
							//return errors;
																						
							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : validation)
								response.add(rbf.present(x));
							
							double[] r = new double[]{
											rbf.scale,
											Meuse.getRMSE(response, validationDesired), 
											Math.pow(Meuse.getPearson(response, validationDesired), 2),
											rbf.getNeurons().size()
									};
									
							return r;
						}
					}));
				}
			}
		
			es.shutdown();

			DescriptiveStatistics[] ds = null;

			for (Future<double[]> f : futures) {
				try {
					double[] d = f.get();

					if (ds == null) {
						ds = new DescriptiveStatistics[d.length];
						for (int i = 0; i < d.length; i++)
							ds[i] = new DescriptiveStatistics();
					}

					for (int i = 0; i < d.length; i++)
						ds[i].addValue(d[i]);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			
			String desc = aMode+","+exp+","+lrA+","+idw+","+ins_per+","+delta+","+a_max+","+t_max+","+n+","+sc+","+adapt_per;
			
			double cor = ds[ds.length-2].getMean();
			if( sc == 0.0 )
				noAdaptCor = cor;
			
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < ds.length; i++)
				sb.append(ds[i].getMean() + ",");
			log.debug(desc+","+sb.substring(0, Math.min(sb.length(),400))+( cor - noAdaptCor));
					
			results.put(desc, ds);
						
		}
		}
		
		FileWriter fw = null;
		try {
			fw = new FileWriter("output/error.csv");
			fw.write("iteration");
			
			List<String> l = new ArrayList<String>(results.keySet());
			
			for(String s : l) {
				log.debug(l.indexOf(s)+"->"+s);
				fw.write("," + l.indexOf(s));
			}
			fw.write("\n");

			int length = results.values().iterator().next().length;
			for (int i = 0; i < length; i++) {
				fw.write(i + "");
				for (String s : l)
					fw.write("," + results.get(s)[i].getMean());
				fw.write("\n");
			}
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static double getRMSE( RBF rbf, List<double[]> validation, List<double[]> validationDesired ) {
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : validation)
			response.add(rbf.present(x));
		return Meuse.getRMSE(response, validationDesired);
	}
}
