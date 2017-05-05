package gwr;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.vividsolutions.jts.geom.Point;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class GWR_CV {

	private static Logger log = Logger.getLogger(GWR_CV.class);

	public static void main(String[] args) {

		int threads = 5;//Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

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

		// lm
		LinearModel lm = new LinearModel(sdf.samples, fa, ta, false);
		List<Double> residuals = lm.getResiduals();
		Drawer.geoDrawValues(sdf.geoms, residuals, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, "output/lm_residuals.png");

		double mean = 0;
		Map<double[], Double> values = new HashMap<>();
		for (int i = 0; i < sdf.samples.size(); i++) {
			values.put(sdf.samples.get(i), residuals.get(i));
			mean += residuals.get(i);
		}
		mean /= residuals.size();

		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);
		Map<double[], Map<double[], Double>> dMap = GeoUtils.contiguityMapToDistanceMap(cm);
		List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo(sdf.samples, values, dMap, 999);
		
		GWKernel k = GWKernel.gaussian;
		boolean adaptive = true;

		List<double[]> samples = sdf.samples;
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 32, samples.size());

		for (double bw = 5; bw < 15; bw++ ) {

			Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);

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
								double w = GeoUtils.getKernelValue(k, d, bandwidth.get(a));
								
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
