package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import cern.colt.Arrays;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering_GWR {

	private static Logger log = Logger.getLogger(ChowClustering_GWR.class);

	public static int STRUCT_TEST = 0, P_VALUE = 1, DIST = 2, MIN_OBS = 3, PRECLUST = 4, PRECLUST_OPT = 5, PRECLUST_OPT2 = 6;

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
		log.debug("Threads: " + threads);

		File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[] { 1, 2 }, true);
		
		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 7, 8, 9, 10, 19, 20 };
		int ta = 18; // bldUpRt

		for (int i = 0; i < fa.length; i++)
			log.debug("fa " + i + ": " + sdf.names.get(fa[i]));
		log.debug("ta: " + ta + "," + sdf.names.get(ta));
				
		// wcm for moran
		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);
		Map<double[],Map<double[],Double>> wcm = GeoUtils.contiguityMapToDistanceMap(cm);
		//Map<double[],Map<double[],Double>> wcm = GeoUtils.getInverseDistanceMatrix(sdf.samples, new GeometryDist(sdf.samples,sdf.geoms), 2);
		GeoUtils.rowNormalizeMatrix(wcm);
		
		// distance for GWR
		Dist<double[]> gDist = new EuclideanDist(ga);

		{
			LinearModel lm = new LinearModel(sdf.samples, fa, ta, false);
			List<Double> pred = lm.getPredictions(sdf.samples, fa);
			double mse = SupervisedUtils.getMSE(pred, sdf.samples, ta);
			log.debug("lm:");
			log.debug("RSS: " + SupervisedUtils.getResidualSumOfSquares(pred, sdf.samples, ta));
			log.debug("bic: " + SupervisedUtils.getBIC(mse, fa.length + 1, sdf.samples.size()));
			log.debug("aicc: " + SupervisedUtils.getAICc_GWMODEL(mse, fa.length + 1, sdf.samples.size())); // lm aic: -61856.98209268832
		}

		DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(sdf.samples, ta));
		DoubleMatrix X = new DoubleMatrix(LinearModel.getX(sdf.samples, fa, true));
		
		class GWR_loc {
			double[] coefficients;
			Double response;
			double[] S_row;
			Double other = 0.0;
		}

		boolean gaussian = true;
		boolean adaptive = true;

		for (int bw = 8; bw <= 30; bw++) {
			final int bandwidth = bw;
			List<Future<GWR_loc>> futures = new ArrayList<>();
			
			ExecutorService es = Executors.newFixedThreadPool(threads);
			//ThreadPoolExecutor es = new ThreadPoolExecutor(threads, threads*2, 20, TimeUnit.HOURS, new ArrayBlockingQueue<Runnable>(20));
			//ThreadPoolExecutor es = new ThreadPoolExecutor(threads, 2*threads, 5000L, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(20, true), new ThreadPoolExecutor.CallerRunsPolicy());
			
			for (int i = 0; i < sdf.samples.size(); i++) {
				final int I = i;

				futures.add(es.submit(new Callable<GWR_loc>() {
					@Override
					public GWR_loc call() throws Exception {
						GWR_loc gl = new GWR_loc();
						double[] a = sdf.samples.get(I);

						double h;
						if (!adaptive)
							h = bandwidth;
						else {
							int k = (int) bandwidth;
							List<double[]> s = new ArrayList<>(sdf.samples);
							Collections.sort(s, new Comparator<double[]>() {
								@Override
								public int compare(double[] o1, double[] o2) {
									return Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
								}
							});
							h = gDist.dist(s.get(k - 1), a);
						}

						DoubleMatrix XtW = new DoubleMatrix(X.getColumns(),X.getRows());		
						for( int j = 0; j < X.getRows(); j++ ) {
							double[] b = sdf.samples.get(j);
							double d = gDist.dist( a, b);
													
							double w;
							if( gaussian ) // Gaussian
								w = Math.exp(-0.5*Math.pow(d/h,2));
							else // bisquare
								w = Math.pow(1.0-Math.pow(d/h, 2), 2);
							XtW.putColumn(j, X.getRow(j).mul(w));
						}	
						
						DoubleMatrix XtWX = XtW.mmul(X);

						DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
						DoubleMatrix rowI = X.getRow(I);

						gl.coefficients = beta.data;
						gl.response = rowI.mmul(beta).get(0);
						gl.S_row = rowI.mmul(Solve.pinv(XtWX)).mmul(XtW).data;

						return gl;
					}
				}));
			}
			es.shutdown();

			List<double[]> coefficients = new ArrayList<>();
			List<Double> response = new ArrayList<>();
			Map<double[],Double> residuals = new HashMap<>();
			List<Double> other = new ArrayList<>();
			double[][] s = new double[futures.size()][];

			for (int i = 0; i < futures.size(); i++) {
				try {
					GWR_loc gl = futures.get(i).get();
					coefficients.add(gl.coefficients);
					response.add(gl.response);
					residuals.put(sdf.samples.get(i),sdf.samples.get(i)[ta]-gl.response);
					other.add(gl.other);
					s[i] = gl.S_row;
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}

			DoubleMatrix S = new DoubleMatrix(s);
			double traceS = S.diag().sum();
			double rss = SupervisedUtils.getResidualSumOfSquares(response, sdf.samples, ta);
			double mse = rss / sdf.samples.size();

			log.debug("Bandwidth: "+bw);
			log.debug("AICc: " + SupervisedUtils.getAICc_GWMODEL(mse, traceS, sdf.samples.size()));
			log.debug("BIC: " + SupervisedUtils.getBIC(mse, traceS, sdf.samples.size()));
			log.debug("R2: " + SupervisedUtils.getR2(response, sdf.samples, ta));
			log.debug("Moran's I: "+Arrays.toString( GeoUtils.getMoransIStatistics(wcm, residuals) ) );
		}
	}
}
