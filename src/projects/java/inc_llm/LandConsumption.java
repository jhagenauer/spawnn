package inc_llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.vividsolutions.jts.geom.Point;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.DataUtils.Transform;

public class LandConsumption {

	private static Logger log = Logger.getLogger(LandConsumption.class);

	public static void main(String[] args) {

		Random r = new Random();
		int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"),
				true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}
		
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 52, 49, 10 };
		int ta = 7;
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);
		
		boolean gaussian = true;
		boolean adaptive = true;
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
				
		List<double[]> params = new ArrayList<>();
		for( double lrB : new double[]{ 0.2, 0.1, 0.05, 0.01, 0.005, 0.001 } )
			for( double lrBln : new double[]{ 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001 } )
				for( double lrN : new double[]{ 0.2, 0.1, 0.05, 0.01, 0.005, 0.001 } )
					for( double lrNln : new double[]{ 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001 } )
						for( double aMax : new double[]{ 50, 100, 200, 400, 800, 1600, 2400 } )
							for( double lambda : new double[]{ 50, 100, 200, 400, 800, 1600, 2400 } ) {
								if( lrB < lrN || lrBln < lrNln )
									continue;
								params.add( new double[]{ lrB, lrBln, lrN, lrNln, aMax, lambda } );
							}
		Collections.shuffle(params);
		log.debug(params.size());
		
		double bestRMSE = Double.POSITIVE_INFINITY;
		double[] bestP = null;
		
		for( double[] p : params ){
			int t_max = 100000;
			int initNeurons = 2;
			
			double lrB = p[0];
			double lrBln = p[1];
			
			double lrN = p[2];
			double lrNln = p[3];
			
			int aMax = (int)p[4];
			int lambda = (int)p[5];
			
			double alpha = 0.5;
			double beta = 0.000005;
			
			List<Future<Double>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {

				futures.add(es.submit(new Callable<Double>() {
					@Override
					public Double call() throws Exception {
				
						List<double[]> samplesTrain = new ArrayList<>();
						List<double[]> desiredTrain = new ArrayList<>();
						for (int k : cvEntry.getKey()) {
							samplesTrain.add(samples.get(k));
							desiredTrain.add( new double[]{samples.get(k)[ta]});
						}
							
						List<double[]> samplesVal = new ArrayList<>();
						List<double[]> desiredVal = new ArrayList<>();
						for (int k : cvEntry.getValue()) {
							samplesVal.add(samples.get(k));
							desiredVal.add( new double[]{samples.get(k)[ta]});
						}
													
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < initNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}
					
						Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);
						
						IncLLM llm = new IncLLM(neurons, lrB, lrBln, lrN, lrNln, sorter, aMax, lambda, alpha, beta, fa, 1);
						int t = 0;
						for (; t < t_max; t++) {
							int idx = r.nextInt(samplesTrain.size());
							llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx) );
						}
						
						List<double[]> responseVal = new ArrayList<double[]>();
						for( int i = 0; i < samplesVal.size(); i++ )
							responseVal.add( llm.present(samplesVal.get(i)));
								
						return SupervisedUtils.getRMSE(responseVal,desiredVal);
					}
				}));
			}			
			es.shutdown();
			SummaryStatistics ss = new SummaryStatistics();	
			try {
				for( Future<Double> f : futures )
					ss.addValue(f.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			if( bestP == null || bestRMSE < ss.getMean() ) {
				bestP = p;
				log.info(Arrays.toString(p)+","+ss.getMean());
			}
		}
		System.exit(1);

		for (double bw = 5; bw < 12; bw++ ) {

			Map<double[], Double> bandwidth = new HashMap<>();
			for (double[] a : samples) {
				if (!adaptive)
					bandwidth.put(a, bw);
				else {
					int k = (int) bw;
					List<double[]> s = new ArrayList<>(samples);
					Collections.sort(s, new Comparator<double[]>() {
						@Override
						public int compare(double[] o1, double[] o2) {
							return Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
						}
					});			
					bandwidth.put(a, gDist.dist(s.get(k - 1), a));
				}
			}

			List<Future<Double>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {

				futures.add(es.submit(new Callable<Double>() {
					@Override
					public Double call() throws Exception {

						List<double[]> samplesTrain = new ArrayList<double[]>();
						for (int k : cvEntry.getKey())
							samplesTrain.add(samples.get(k));

						List<double[]> samplesVal = new ArrayList<double[]>();
						for (int k : cvEntry.getValue())
							samplesVal.add(samples.get(k));

						DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samplesTrain, ta));
						DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samplesTrain, fa, true));

						DoubleMatrix XVal = new DoubleMatrix(LinearModel.getX(samplesVal, fa, true));
						List<Double> predictions = new ArrayList<>();
						for (int i = 0; i < samplesVal.size(); i++) {
							double[] a = samplesVal.get(i);

							DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
							for (int j = 0; j < X.getRows(); j++) {
								double[] b = samplesTrain.get(j);
								double d = gDist.dist(a, b);

								double w;
								if (gaussian) // Gaussian
									w = Math.exp(-0.5 * Math.pow(d / bandwidth.get(a), 2));
								else // bisquare
									w = Math.pow(1.0 - Math.pow(d / bandwidth.get(a), 2), 2);
								XtW.putColumn(j, X.getRow(j).mul(w));
							}
							DoubleMatrix XtWX = XtW.mmul(X);
							DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));

							predictions.add(XVal.getRow(i).mmul(beta).get(0));
						}
						return SupervisedUtils.getRMSE(predictions, samplesVal, ta);
					}
				}));
			}
			
			es.shutdown();
			SummaryStatistics ss = new SummaryStatistics();	
			try {
				for( Future<Double> f : futures )
					ss.addValue(f.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			log.debug(bw+","+ss.getMean());
		}
	}
}
