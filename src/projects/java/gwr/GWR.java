package gwr;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
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

import chowClustering.ChowClustering;
import chowClustering.ChowClustering.StructChangeTestMode;
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
								
		int[] fa = new int[]{52,49,10};
		int ta = 7;
		
		{ // lm
			LinearModel lm = new LinearModel(sdf.samples, fa, ta, false);
			List<Double> residuals = lm.getResiduals();			
			Drawer.geoDrawValues(sdf.geoms, residuals, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, "output/lm_residuals.png");
		}
		
		boolean gaussian = true;
		boolean adaptive = true;
		boolean trueAdaptive = false;
		double bandwidth = 19;
		
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( sdf.samples, ta) );
		DoubleMatrix X = new DoubleMatrix( LinearModel.getX( sdf.samples, fa, true) );
		DoubleMatrix Xt = X.transpose();
		
		class GWR_loc {
			double[] coefficients;
			Double response;
			double[] S_row;
			Double other = 0.0;
		}
		
		Path file = Paths.get("output/gwr_test.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
			String s = "id,i,pValue\r\n";
			Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		long time = System.currentTimeMillis();
		
		List<Future<GWR_loc>> futures = new ArrayList<>();
		ExecutorService es = Executors.newFixedThreadPool(threads);
		
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			final int I = i;
			
			futures.add(es.submit(new Callable<GWR_loc>() {
				@Override
				public GWR_loc call() throws Exception {
					GWR_loc gl = new GWR_loc();
					double[] a = sdf.samples.get(I);
					
					double h;
					if( !adaptive )
						h = bandwidth;
					else {
						int k = (int)bandwidth;
						List<double[]> s = new ArrayList<>(sdf.samples);
						Collections.sort(s, new Comparator<double[]>() {
							@Override
							public int compare(double[] o1, double[] o2) { 
								return Double.compare( gDist.dist(o1, a), gDist.dist(o2, a)); 
							}
						});
						h = gDist.dist( s.get( k-1 ), a);
						
						if( trueAdaptive ) {							
							double[][] x1 = LinearModel.getX(s.subList(0, k), fa, true);
							double[] y1 = LinearModel.getY(s.subList(0, k), ta);
							
							for( int i = k+1; i < s.size(); i++ ) {
								double[][] x2 = LinearModel.getX(s.subList(0, i), fa, true);
								double[] y2 = LinearModel.getY(s.subList(0, i), ta);
								double[] r = ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.AdjustedChow);
								
								/*try {
									String st = i+","+r[0]+","+r[1]+"\r\n";
									Files.write(file, st.getBytes(), StandardOpenOption.APPEND);
								} catch (IOException e) {
									e.printStackTrace();
								}*/
								
								//log.debug(i+","+Arrays.toString(r));
								if( r[1] < 0.5 ) {
									h = gDist.dist( s.get( i-1 ), a);
									gl.other = (double)(i-1);
									log.debug(I+","+i+","+Arrays.toString(r));
									break;
								}
							}
							
							//System.exit(1);
						}
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
					
					DoubleMatrix XtW = Xt.mmul(new DoubleMatrix(w));
					DoubleMatrix XtWX = XtW.mmul(X);
					
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
					DoubleMatrix rowI = X.getRow(I);
					
					gl.coefficients = beta.data;
					gl.response = rowI.mmul( beta ).get(0);
					gl.S_row = rowI.mmul( Solve.pinv(XtWX) ).mmul(XtW).data;
															
					return gl;
			}}));
		}
		es.shutdown();
		
		
		List<double[]> coefficients = new ArrayList<>();
		List<Double> response = new ArrayList<Double>();
		List<Double> other = new ArrayList<>();
		double[][] s = new double[futures.size()][];
				
		for( int i = 0; i < futures.size(); i++ ) {
			try {
				GWR_loc gl = futures.get(i).get();
				coefficients.add(gl.coefficients);
				response.add(gl.response);
				other.add(gl.other);
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
		
		Drawer.geoDrawValues(sdf.geoms, other, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, "output/other.png");
		
		SummaryStatistics ss = new SummaryStatistics();
		for( double o : other )
			ss.addValue(o);
		log.debug("mean other: "+ss.getMean() );
	}
}
