package gwr;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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

import nnet.SupervisedUtils;
import regioClust.LinearModel;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWR_CV {

	private static Logger log = Logger.getLogger(GWR_CV.class);

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 52, 49, 10 };
		int ta = 7;

		//DataUtils.transform(sdf.samples, fa, Transform.zScore);

		Dist<double[]> gDist = new EuclideanDist(ga);

		List<Object[]> params = new ArrayList<Object[]>();
		for( int bw = 14; bw < 32; bw++ )
				params.add( new Object[]{ GWKernel.boxcar, true, (double)bw } );
		for( int bw = 12; bw < 32; bw++ )
			params.add( new Object[]{ GWKernel.bisquare, true, (double)bw } );	
		for( int bw = 3; bw < 32; bw++ )
			params.add( new Object[]{ GWKernel.gaussian, true, (double)bw } );		
		for (double bw = 0.4; bw < 4; bw += 0.1)
			params.add( new Object[]{ GWKernel.gaussian, false, bw } );
		Collections.shuffle(params);

		List<double[]> samples = sdf.samples;
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		
		double bestRMSE = Double.MAX_VALUE;
		for (Object[] p : params ) {
			log.debug(Arrays.toString(p));
			Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, (Double)p[2], (Boolean)p[1] );

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
								double w = GeoUtils.getKernelValue( (GWKernel)p[0], d, bandwidth.get(a));

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
				for (Future<Double> f : futures)
					ss.addValue(f.get());
			} catch (InterruptedException e) {
				log.error(Arrays.toString(p));
				e.printStackTrace();
			} catch (ExecutionException e) {
				log.error(Arrays.toString(p));
				e.printStackTrace();
			}
			if( ss.getMean() < bestRMSE ) {
				log.info( Arrays.toString(p) + "," + ss.getMean());
				bestRMSE = ss.getMean();
			}
		}
	}
}
