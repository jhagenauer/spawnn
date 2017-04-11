package gwr;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
				
		int[] ga = new int[]{0,1};					
		int[] fa = new int[]{52,49,10};
		int ta = 7;
		
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		
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
		
		/*List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, sdf.samples.size() );
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			for (int k : cvEntry.getKey()) 
				samplesTrain.add(sdf.samples.get(k));
				
			List<double[]> samplesVal = new ArrayList<double[]>();
			for (int k : cvEntry.getValue()) 
				samplesVal.add(sdf.samples.get(k));
		}*/
		
		List<double[]> samples = sdf.samples;
		List<double[]> samplesTrain = new ArrayList<double[]>();
		List<double[]> samplesVal = new ArrayList<double[]>();
		for( int k = 0; k < samples.size(); k++ ) {
			if( k < 3000 )
				samplesTrain.add( samples.get(k));
			else
				samplesVal.add( samples.get(k));		
		}
		
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( samplesTrain, ta) );
		DoubleMatrix X = new DoubleMatrix( LinearModel.getX( samplesTrain, fa, true) );
								
		for( double bandwidth : new double[]{ 9 } ) {

			log.debug("GWR");
			log.debug("bandwidth: "+bandwidth);
												
			// estimate coefficients
			List<Double> responseTrain = new ArrayList<Double>();
			DoubleMatrix betas = new DoubleMatrix(samplesTrain.size(),fa.length+1);
			DoubleMatrix S = new DoubleMatrix(samplesTrain.size(),samplesTrain.size());
			
			for( int i = 0; i < samplesTrain.size(); i++ ) {
				double[] a = samplesTrain.get(i);
				
				if( 1 != 0 )
				break;
				
				// calculate bandwidths
				double h;
				if( !adaptive )
					h = bandwidth;
				else {
					int k = (int)bandwidth;
					List<double[]> s = new ArrayList<>(samples); // all or just train?
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
					double[] b = samplesTrain.get(j);
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
				DoubleMatrix rowI = X.getRow(i);
				
				betas.putRow(i, beta);
				responseTrain.add( rowI.mmul(beta).get(0) );
				S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );												
			}
			
			DoubleMatrix XVal = new DoubleMatrix( LinearModel.getX( samplesVal, fa, true) );
						
			for( int i = 0; i < samplesVal.size(); i++ ) {
				double[] a = samplesVal.get(i);
				
				// calculate bandwidths
				double h;
				if( !adaptive )
					h = bandwidth;
				else {
					int k = (int)bandwidth;
					List<double[]> s = new ArrayList<>(samples); // all or just train?
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
					double[] b = samplesTrain.get(j);
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
				
				double prediction = XVal.getRow(i).mmul(beta).get(0);
			}
			
			double rss = SupervisedUtils.getResidualSumOfSquares(responseTrain, samplesTrain, ta);
			double mse = rss/samplesTrain.size();
			double traceS = S.diag().sum();
			
			log.debug("Nr params: "+traceS);		
			log.debug("Nr. of data points: "+samplesTrain.size());
			/*double traceStS = S.transpose().mmul(S).diag().sum();
			log.debug("Effective nr. Params: "+(2*traceS-traceStS));
			log.debug("Effective degrees of freedom: "+(samplesTrain.size()-2*traceS+traceStS));*/
			log.debug("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, traceS, samplesTrain.size())); 
			log.debug("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, traceS, samplesTrain.size())); 
			log.debug("RSS: "+rss); 
			log.debug("R2: "+SupervisedUtils.getR2(responseTrain, samplesTrain, ta));  
			
		}
	}
}
