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
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class Ontario2 {

	private static Logger log = Logger.getLogger(Ontario2.class);
	
	public static void main(String[] args) {
		final Random r = new Random();
		
		SpatialDataFrame sd = DataUtils.readSpatialDataFrameFromShapefile(new File("data/ontario/clipped/ontario_inorg_sel_final.shp"), true);
		Map<String, DescriptiveStatistics[]> results = new HashMap<String, DescriptiveStatistics[]>();
							
		double noAdaptCor = 0;
						
		//for( int n = 7; n <= 77; n++ ) 
		for( int n : new int[]{ 33 } ) // co (14), ni (33), zn (55), al (57), V (51)
		for( int nnSize : new int[]{ 0, 7 } ) {
				
		EuclideanDist dist23 = new EuclideanDist(new int[] { 2, 3 });
		List<double[]> all = new ArrayList<double[]>();
		for (double[] d : sd.samples ) {
						
			List<double[]> nns = new ArrayList<double[]>();
			while( nns.size() < nnSize ) {
				
				double[] nn = null;
				for( double[] d2 : sd.samples ) {
					if( d == d2 || nns.contains(d2) )
						continue;
					
					if( nn == null || dist23.dist(d, d2) < dist23.dist( nn, d)  )
						nn = d2;
				}
				nns.add(nn);
			}
			
			double[] d3 = new double[3+nns.size()];
			d3[0] = d[2]; // x
			d3[1] = d[3]; // y
			d3[2] = d[n]; // n (target)
			for( int i = 0; i < nns.size(); i++ )
				d3[i+3] = nns.get(i)[n];
			all.add( d3 );
		}
					
		int[] ga = new int[]{ 0, 1 };
		DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga) );
		
		int[] fa1 = new int[all.get(0).length-3];
		for( int i = 0; i < fa1.length; i++ )
			fa1[i] = i+3;	
		DataUtils.zScoreColumns(all, fa1 );
		
		int fa2 = 2;
		//DataUtils.zScoreColumn(all, fa2);
		/*for( double[] d : all )
			d[fa2] = Math.sqrt(d[fa2]);*/
				
		final Dist<double[]> dist = new EuclideanDist();
		
		int threads = 4;
		int maxRuns = 16;
		final int maxK = 10;
				
		for( final int aMode : new int []{ 1 } ) 
		for( int t_max : new int[]{ 100000 } )
		for( final int ins_per : new int[]{ t_max/10 } )
		for( final double delta : new double[]{ 0.1 } )
		for( final int a_max : new int[]{ 100 } )
		for( final double lrA : new double[]   { 0.05 }  ) // movement bmu
		for( final double lrB : new double[]   { 0.0005 }  ) // movement neighbors
		for( final double alpha : new double[] { 0.05 } ) // all error reduction
		for( final double beta : new double[] { 0.5 } ) // error reduction due to insertion
		for( double mod : new double[]{ 0.0 /*, 0.5*/ } )
		for( final int adapt_per : new int[]{ 1000 } ) {
			
			if( mod == 0.0 && adapt_per > 1000 )
				continue;
									
			final double MOD = mod;
			final int T_MAX = t_max;
												
			ExecutorService es = Executors.newFixedThreadPool( threads );
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
			for (int j = 0; j < maxRuns; j++) {
				Collections.shuffle(all);
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();

				for( double[] d : all ) {
					
					double[] n2 = new double[ga.length+fa1.length];
					for( int i = 0; i < ga.length; i++ )
						n2[i] = d[ga[i]];
					for( int i = 0; i < fa1.length; i++ )
						n2[i+ga.length] = d[fa1[i]];
										
					samples.add(n2);
					desired.add(new double[] { d[2] });
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
							
							double[] errors = new double[T_MAX];

							AdaptIncRBF rbf = new AdaptIncRBF(hidden, lrA, lrB, dist, a_max, MOD, alpha, beta, delta, 1);
							
							double bestError = Double.MAX_VALUE;
							double bestT = 0;
							
							for (int t = 0; t < T_MAX; t++) {
								int idx = r.nextInt(training.size());																
								rbf.train(training.get(idx), trainingDesired.get(idx));		
															
								if( t % ins_per == 0 ) // 0.46608810622495844
									rbf.insert();
								else if( t % adapt_per == 0 ) {				
									if( aMode == 0 )
										rbf.adaptScale( rbf.getTotalError() ); // conventional
									else 
										rbf.adaptScale( Ontario.getRMSE(rbf, training, trainingDesired) );
								}
								
								/*int d = T_MAX / errors.length;
								if (t % d == 0) {
									List<double[]> response = new ArrayList<double[]>();
									for (double[] x : validation)
										response.add(rbf.present(x));
									errors[t / d] = Meuse.getRMSE(response, validationDesired);
								}*/
								
								// better than fixed insertion?, 0.4669724360759605
								/*double e = rbf.getTotalError();
								if( e < bestError ) {
									bestError = e;
									bestT = t;
								} else if( t - bestT > ins_per && rbf.getNeurons().size() < 50 ) {
									rbf.insert();
									bestT = t; // give it some time
									bestError = e;						
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
			
			String desc = aMode+","+nnSize+","+lrA+","+lrB+","+ins_per+","+delta+","+a_max+","+mod+","+alpha+","+beta+","+t_max+","+n+","+adapt_per;
			
			/*String desc = "";
			if( mod == 0 ) 
				desc += "Basic";
			else
				desc += "Adaptive";
			
			if( nnSize > 0 )
				desc += " with NBs";*/
			
			double cor = ds[ds.length-2].getMean();
			if( mod == 0.0 )
				noAdaptCor = cor;
			
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < ds.length; i++)
				sb.append(ds[i].getMean() + ",");
			log.debug(desc+","+sb.substring(0, Math.min(sb.length(),500))+( cor - noAdaptCor));
					
			results.put(desc, ds);
						
		}
		}
		
		FileWriter fw = null;
		try {
			fw = new FileWriter("output/error.csv");
			fw.write("iteration");
			
			List<String> keys = new ArrayList<String>(results.keySet());
			
			
			for(String k : keys) 
				fw.write("," + k);
			fw.write("\n");

			int length = results.get(keys.get(0)).length;
			for (int i = 0; i < length; i++) {
				fw.write(i + "");
				for (String s : keys)
					fw.write("," + results.get(s)[i].getMean());
				fw.write("\n");
			}
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
