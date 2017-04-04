package gwr;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

public class GWR {
	
	private static Logger log = Logger.getLogger(GWR.class);

	public static void main(String[] args) {
		int threads = Math.max(1, Runtime.getRuntime().availableProcessors()-1);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}
		Dist<double[]> gDist = new EuclideanDist(new int[]{0,1});
		
		sdf.geoms = sdf.geoms.subList(0, 1000);
		sdf.samples = sdf.samples.subList(0, 1000);
				
		int[] fa = new int[]{52,49,10};
		int ta = 7;
		
		boolean gaussian = true;
		boolean adaptive = true;
		double bandwidth = 22;
		
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( sdf.samples, ta) );
		DoubleMatrix X = new DoubleMatrix( LinearModel.getX( sdf.samples, fa, true) );
		DoubleMatrix Xt = X.transpose();
		
		class GWR_loc {
			double[] coefficients;
			Double response;
			double[] S_row;
		}
		
		long time = System.currentTimeMillis();
		
		List<Future<GWR_loc>> futures = new ArrayList<>();
		ExecutorService es = Executors.newFixedThreadPool(threads);
		
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			final int I = i;
			
			futures.add(es.submit(new Callable<GWR_loc>() {
				@Override
				public GWR_loc call() throws Exception {
					double[] a = sdf.samples.get(I);
					
					double h;
					if( !adaptive )
						h = bandwidth;
					else {
						List<double[]> s = new ArrayList<>(sdf.samples);
						Collections.sort(s, new Comparator<double[]>() {
							@Override
							public int compare(double[] o1, double[] o2) { 
								return Double.compare( gDist.dist(o1, a), gDist.dist(o2, a)); 
							}
						});
						h = gDist.dist( s.get( (int)bandwidth-1 ), a);
					}
								
					double[][] w = new double[sdf.samples.size()][sdf.samples.size()];						
					for( int j = 0; j < sdf.samples.size(); j++ ) {
						double[] b = sdf.samples.get(j);
						
						double d = gDist.dist( a, b);
						
						// Gaussian
						if( gaussian )
							w[j][j] = Math.exp(-0.5*Math.pow(d/h,2));
						else // bisquare
							w[j][j] = Math.pow(1.0-Math.pow(d/h, 2), 2);
					}	
					DoubleMatrix W = new DoubleMatrix(w);
					
					DoubleMatrix XtW = Xt.mmul(W);
					DoubleMatrix XtWX = XtW.mmul(X);
					
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
					DoubleMatrix rowI = X.getRow(I);
					
					GWR_loc gl = new GWR_loc();
					gl.coefficients = beta.data;
					gl.response = rowI.mmul( beta ).get(0);
					gl.S_row = rowI.mmul( Solve.pinv(XtWX) ).mmul(XtW).data;
															
					return gl;
			}}));
		}
		es.shutdown();
		
		
		List<double[]> coefficients = new ArrayList<>();
		List<Double> response = new ArrayList<Double>();
		double[][] s = new double[futures.size()][];
				
		for( int i = 0; i < futures.size(); i++ ) {
			try {
				GWR_loc gl = futures.get(i).get();
				coefficients.add(gl.coefficients);
				response.add(gl.response);
				s[i] = gl.S_row;
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		log.debug("took: "+(System.currentTimeMillis()-time)/1000.0);
						
		DoubleMatrix S = new DoubleMatrix(s);
		double traceS = S.diag().sum();
		double traceStS = S.transpose().mmul(S).diag().sum();
		double rss = SupervisedUtils.getResidualSumOfSquares(response, sdf.samples, ta);
		double mse = rss/sdf.samples.size();
		
		log.debug("Nr params: "+traceS);		
		log.debug("Nr. of data points: "+sdf.samples.size());
		log.debug("Effective nr. Params: "+(2*traceS-traceStS));
		log.debug("Effective degrees of freedom: "+(sdf.samples.size()-2*traceS+traceStS));
		log.debug("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, traceS, sdf.samples.size())); 
		log.debug("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, traceS, sdf.samples.size())); 
		log.debug("RSS: "+rss); 
		log.debug("R2: "+SupervisedUtils.getR2(response, sdf.samples, ta));  
						
		for( int i = 0; i < fa.length; i++ )
			Drawer.geoDrawValues(sdf.geoms, coefficients, i, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, "output/coef_"+sdf.names.get(fa[i])+".png");
	}
}
