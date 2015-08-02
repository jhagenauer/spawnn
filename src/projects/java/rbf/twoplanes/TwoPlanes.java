package rbf.twoplanes;

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
import rbf.ontario.Ontario;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.AdaptIncRBF;
import spawnn.utils.DataUtils;

public class TwoPlanes {

	private static Logger log = Logger.getLogger(TwoPlanes.class);
	
	public static void main(String[] args) {
		final Random r = new Random();

		List<double[]> all = DataUtils.readCSV("data/twoplanes.csv");
								
		final Dist<double[]> dist = new EuclideanDist();
		final int maxK = 10;
		double noAdaptCor = 0;
		
		// aMode 1, best:
		// 1,0.05,0.001,0.0,0.05,400,0.1|-0.5063112740812139,0.4746422663321312,0.6823850653758061,-1.8890785267374648,0.45620450108619304,0.47541038196231267,0.6823850653758061,1.8903291644329405,0.7072,0.9783792449999811,0.4067431294330523,0.8393406035969778,
		// sd radius: 0.2756711725855279, sd rmse: 0.048815044319219944, sd cor 0.03687566403640481: 
		
		for( final int aMode : new int[]{ 1 } )
		for( final int T_MAX : new int[] { 10000 } )
		for( final double lrA : new double[]{ 0.05, 0.02 } )
		for( final double lrB : new double[]{ /*0.002,*/ 0.001, 0.0005 } )
		for( final double alpha : new double[] { 0.0003 } )
		for( final double delta : new double[]{ 0.05 } )
		for( final double sc : new double[]{ 0.0, 0.1 /*, 0.05*/ } ) 	
		for( final int adapt_per : new int[]{ 250, 400 /*, 800, 1000*/ } ) {
			
			if( sc == 0.0 && adapt_per > 250 )
				continue;
											
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int j = 0; j < 1000; j++) {
					
				Collections.shuffle(all);
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();

				for( double[] d : all ) {
					samples.add(new double[] { d[0], d[1] });	
					desired.add(new double[] { d[2] });
				}
				
				final int nrOut = desired.get(0).length;
				
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
														
							AdaptIncRBF rbf = new AdaptIncRBF(hidden, lrA, lrB, dist, T_MAX*2, sc, alpha, 0.0, delta, nrOut);
							
							int count = 0;
							for (int t = 0; t < T_MAX; t++) {
								int idx = r.nextInt(training.size());																
								rbf.train(training.get(idx), trainingDesired.get(idx));	
								
								if( t % adapt_per == 0 ) {
									
									if( aMode == 0 )
										rbf.adaptScale( rbf.getTotalError() ); // conventional
									else 
										rbf.adaptScale( Ontario.getRMSE(rbf, training, trainingDesired) ); 
								}
							}

							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : validation)
								response.add(rbf.present(x));
														
							List<double[]> ns = new ArrayList<double[]>(rbf.getNeurons().keySet());
							if( ns.get(0)[0] > ns.get(1)[0] ) // sort by x coordinate 
								Collections.reverse(ns);
														
							double[] cm = getConfusionMatrix(response, validationDesired);
							double precision = cm[0] / (cm[0] + cm[2]); // tp durch alle p
							double recall = cm[0] / (cm[0] + cm[1]); // tp durch all true
							double fmeasure = 2 * ( precision * recall ) / ( precision + recall );
							double sf = ((AdaptIncRBF)rbf).scale;
																					
							return append( 
									ns.get(0), 
									new double[]{ rbf.getNeurons().get(ns.get(0) ) }, // radius
									new double[]{rbf.getWeights().get(0).get(ns.get(0) ) }, // weigth
									
									ns.get(1), 
									new double[]{ rbf.getNeurons().get(ns.get(1) ) },
									new double[]{rbf.getWeights().get(0).get(ns.get(1) ) },
									
									new double[]{sf},
									//cm,
									new double[]{fmeasure},
									new double[]{Meuse.getRMSE(response, validationDesired)},
									new double[]{Math.pow(Meuse.getPearson(response, validationDesired), 2)},
									new double[]{count}
								);
						}
					}));
				}
			}

			es.shutdown();

			List<DescriptiveStatistics> ds = null;

			for (Future<double[]> f : futures) {
				try {
					double[] d = f.get();

					if (ds == null) {
						ds = new ArrayList<DescriptiveStatistics>();
						for (int i = 0; i < d.length; i++)
							ds.add(new DescriptiveStatistics());
					}
					for (int i = 0; i < d.length; i++)
						ds.get(i).addValue(d[i]);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			
			String desc = aMode+","+T_MAX+","+lrA+","+lrB+","+alpha+","+delta+","+sc+","+adapt_per;
			
			double cor = ds.get(ds.size()-2).getMean();
			if( sc == 0.0 )
				noAdaptCor = cor;
						
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < ds.size(); i++)
				sb.append(ds.get(i).getMean()+",");//+ "sd: "+ds.get(i).getStandardDeviation()+",");
						
			log.debug(desc+","+sb + ( cor - noAdaptCor));
		}
	}
	
	public static double[] getConfusionMatrix( List<double[]> response, List<double[]> desired ) {
		int tp = 0, fp = 0, tn = 0, fn = 0;
		
		for( int i = 0; i < response.size(); i++ ) {
			
			int r = 0;
			if( response.get(i)[0] > 0 )
				r = 1;
			else if ( response.get(i)[0] < 0 )
				r = -1;
								
			int d = (int) Math.round(desired.get(i)[0] );
			
			if( d == 1 ) { // positive
				if( r == d )
					tp++;
				else 
					fp++;
			} else { // negative
				if( r == d )
					tn++;
				else
					fn++;
			}
		}
		return new double[]{tp,fn,fp,tn};
	}
	
	public static double[] append( double[] ... da ) {
		List<Double> l = new ArrayList<Double>();
		for( double[] d : da )
			for( double d2 : d )
				l.add(d2);
		double[] r = new double[l.size()];
		for( int i = 0; i < l.size(); i++ )
			r[i] = l.get(i);
		return r;
	}
	
}
