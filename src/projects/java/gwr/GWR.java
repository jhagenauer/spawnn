package gwr;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
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
			
			double mean = 0;
			Map<double[],Double> values = new HashMap<>();
			for( int i = 0; i < sdf.samples.size(); i++ ) {
				values.put(sdf.samples.get(i),residuals.get(i));
				mean += residuals.get(i);
			}
			mean /= residuals.size();
			
			Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);
			Map<double[], Map<double[],Double>> dMap =GeoUtils.contiguityMapToDistanceMap(cm);
			List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo(sdf.samples, values, dMap, 999);
			Drawer.geoDrawValues(sdf.geoms, lisa, 0, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, "output/lm_lisa.png");
			
			final Map<Integer, Set<double[]>> lisaCluster = new HashMap<Integer, Set<double[]>>();
			for (int i = 0; i < sdf.samples.size(); i++) {
				double[] l = lisa.get(i);
				double d = residuals.get(i);
				int clust = -1;

				if (l[4] > 0.05) // not significant
					clust = 0;
				else if (l[0] > 0 && d > mean)
					clust = 1; // high-high
				else if (l[0] > 0 && d < mean)
					clust = 2; // low-low
				else if (l[0] < 0 && d > mean)
					clust = 3; // high-low
				else if (l[0] < 0 && d < mean)
					clust = 4; // low-high
				else
					clust = 5; // unknown

				if (!lisaCluster.containsKey(clust))
					lisaCluster.put(clust, new HashSet<double[]>());
				lisaCluster.get(clust).add(sdf.samples.get(i));
			}
			Drawer.geoDrawCluster(lisaCluster.values(), sdf.samples, sdf.geoms, "output/lm_lisa_clust.png", true);
					
			double rss = lm.getRSS();
			double mse = rss/sdf.samples.size();
			log.debug("LM");
			log.debug("Nr params: "+(fa.length+1));		
			log.debug("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, fa.length+1, sdf.samples.size()));
			log.debug("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, fa.length+1, sdf.samples.size()));
			log.debug("RSS: "+rss);
			log.debug("R2: "+SupervisedUtils.getR2(lm.getPredictions(sdf.samples, fa), sdf.samples, ta));
		}
		
		boolean gaussian = true;
		boolean adaptive = true;
		double bandwidth = 19;
		
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( sdf.samples, ta) );
		DoubleMatrix X = new DoubleMatrix( LinearModel.getX( sdf.samples, fa, true) );
		
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
								
		// estimate coefficients
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			final int I = i;
			
			futures.add(es.submit(new Callable<GWR_loc>() {
				@Override
				public GWR_loc call() throws Exception {
					double[] a = sdf.samples.get(I);
					
					// calculate bandwidths
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
					
					GWR_loc gl = new GWR_loc();
					gl.coefficients = beta.data;
					gl.response = rowI.mmul(beta).get(0);
					gl.S_row = rowI.mmul(Solve.pinv(XtWX)).mmul(XtW).data;
															
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
		
		log.debug("GWR");
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
