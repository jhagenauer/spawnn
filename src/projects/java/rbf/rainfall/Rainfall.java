package rbf.rainfall;

import java.io.File;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

// kriging: r^2 = 0.742, rmse = 48.2 

public class Rainfall {

	private static Logger log = Logger.getLogger(Rainfall.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 25000;
		
		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/rainfall_tw/rainfall_tw.csv"), new int[] {}, true);
		Collections.shuffle(df.samples);
		
		for( int knn = 0; knn < 10; knn++ ) {
		//for( int knn : new int[]{ 1 } ) {
		log.debug(knn+":");
		
		int n = 3;
		EuclideanDist dist12 = new EuclideanDist(new int[] { 1, 2 });
		List<double[]> all = new ArrayList<double[]>();
		for (double[] d : df.samples) {

			List<double[]> nns = new ArrayList<double[]>();
			while (nns.size() < knn) {

				double[] nn = null;
				for (double[] d2 : df.samples) {
					if (d == d2 || nns.contains(d2))
						continue;

					if (nn == null || dist12.dist(d, d2) < dist12.dist(nn, d))
						nn = d2;
				}
				nns.add(nn);
			}

			double[] d3 = new double[3 + nns.size()];
			d3[0] = d[1]; // x
			d3[1] = d[2]; // y
			d3[2] = d[n]; // n (target)
			for (int i = 0; i < nns.size(); i++)
				d3[i + 3] = nns.get(i)[n];
			all.add(d3);
		}
		
		int[] ga = new int[] { 0, 1 };
		DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga)); // problem?
		
		int[] fa1 = new int[all.get(0).length - 3];
		for (int i = 0; i < fa1.length; i++)
			fa1[i] = i + 2;
		DataUtils.zScoreColumns(all, fa1);
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : all) {
			samples.add(Arrays.copyOfRange(d, 0, d.length-1));
			desired.add(new double[] { d[d.length-1] });
		}
		
		for( int i = 0; i < samples.size(); i++ )
			;//log.debug( Arrays.toString( samples.get(i) )+","+Arrays.toString( desired.get(i) ) ); 
						
		final Dist<double[]> dist = new EuclideanDist();
		
		final int nrPrototypes = 32;
	
		ExecutorService es = Executors.newFixedThreadPool(4);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		final int maxK = 10;
		
		for (int j = 0; j < 10; j++) {
			
			for (int k = 0; k < maxK; k++) {
				final int K = k;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						List<double[]> training = new ArrayList<double[]>();
						List<double[]> testing = new ArrayList<double[]>();

						for (int i = 0; i < samples.size(); i++)
							if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK)
								testing.add(samples.get(i));
							else
								training.add(samples.get(i));
												
						//Map<double[],Set<double[]>> map = Clustering.kMeans(samples, nrPrototypes, fDist);
						// NG
						Sorter<double[]> s;
						s = new DefaultSorter<double[]>(dist);
						//s = new KangasSorter<double[]>(gDist, fDist, radius);
						NG ng = new NG(nrPrototypes, (double) nrPrototypes / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);
						
						for( int t = 0; t < T_MAX*4; t++ ) 
							ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
						Map<double[],Set<double[]>> map = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);
						
						Map<double[],Double> hidden = new HashMap<double[],Double>();
						// min plus overlap
						for (double[] c : map.keySet() ) {
							double d = Double.MAX_VALUE;
							for (double[] n :  map.keySet() )
								if (c != n)
									d = Math.min(d, dist.dist(c, n))*1.1;
							hidden.put(c, d);
						}
						
						// standard deviation
						/* for (double[] c : map.keySet() ) {
							double sd = 0;
							for( double[] d : map.get(c) )
								sd += Math.pow( fDist.dist(d,c), 2 );
							hidden.put(c, Math.sqrt(sd/map.get(c).size() ) );
						}*/
						
						// min-p
						/*int p = 2;
						for (double[] c : ng.getNeurons() ) {
							
							Set<double[]> cs = new HashSet<double[]>();
							while( cs.size() != p ) {
								double[] closest = null;
								for( double[] d : map.get(c) ) {
									if( cs.contains(d) )
										continue;
									if( closest == null || fDist.dist(d, c) < fDist.dist(closest, c) )
										closest = d;
								}
								cs.add(closest);
							}
							
							double sd = 0;
							for( double[] d : cs )
								sd += Math.pow( fDist.dist(d,c), 2 );
							hidden.put(c, Math.sqrt(sd/map.get(c).size() ) );
						}*/
					

						RBF rbf = new RBF(hidden, 1, dist, 0.05);
						for (int i = 0; i < T_MAX; i++) {
							int j = r.nextInt(samples.size());
							rbf.train(samples.get(j), desired.get(j));
						}
						
						List<double[]> response = new ArrayList<double[]>();
						List<double[]> desiredResponse = new ArrayList<double[]>();
						for (double[] d : testing) {
							response.add(rbf.present(d));
							desiredResponse.add(desired.get(samples.indexOf(d)));
						}
												
						/*double[] cm = getConfusionMatrix(response, desiredResponse);
						double precision = cm[0] / (cm[0] + cm[2]); // tp durch alle p
						double recall = cm[0] / (cm[0] + cm[1]); // tp durch all true
						double fmeasure = 2 * ( precision * recall ) / ( precision + recall );*/

						return new double[] { 
								Meuse.getRMSE(response, desiredResponse), 
								Math.pow(Meuse.getPearson(response, desiredResponse), 2),
								/*cm[0],cm[1],cm[2],cm[3],
								precision,
								recall,
								fmeasure*/
							};
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

		String desc = "";

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < ds.length; i++)
			sb.append(ds[i].getMean() + ",");
		log.debug(desc + "," + sb.substring(0, Math.min(sb.length(), 500)));
		}
	}
	
	public static double[] getConfusionMatrix( List<double[]> response, List<double[]> desired ) {
		int tp = 0, fn = 0, tn = 0, fp = 0;
		
		for( int i = 0; i < response.size(); i++ ) {
			
			int r;
			if( Math.abs( response.get(i)[0] - 0 ) < Math.abs( response.get(i)[0] - 1 ) )
				r = 0;
			else 
				r = 1;
								
			int d = (int) Math.round(desired.get(i)[0] ); // 0 or 1
			
			if( d == 1 ) { // positive
				if( r == d )
					tp++;
				else 
					fn++;
			} else { // negative
				if( r == d )
					tn++;
				else
					fp++;
			}
		}
		return new double[]{tp,fp,fn,tn};
	}

}
