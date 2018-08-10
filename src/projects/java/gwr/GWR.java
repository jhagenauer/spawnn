package gwr;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.vividsolutions.jts.geom.Point;

import nnet.SupervisedUtils;
import regioClust.LinearModel;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWR {
	
	private static Logger log = Logger.getLogger(GWR.class);

	public static void main(String[] args) {
		
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
		Drawer.geoDrawValues(sdf.geoms, lisa, 0, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile,"output/lm_lisa.png");

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
			
		boolean adaptive = true;
				
		List<double[]> samples = sdf.samples;
			
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( samples, ta) );
		DoubleMatrix X = new DoubleMatrix( LinearModel.getX( samples, fa, true) );
		
		for( GWKernel k : new GWKernel[]{ GWKernel.gaussian, GWKernel.bisquare, GWKernel.boxcar } )
		for( double bw : new double[]{ 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 } ) {
			
			// estimate coefficients
			List<Double> responseTrain = new ArrayList<Double>();
			DoubleMatrix betas = new DoubleMatrix(X.getRows(),X.getColumns() );
			DoubleMatrix S = new DoubleMatrix( X.getRows(), X.getRows() );
			
			Map<double[],Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
						
			for( int i = 0; i < samples.size(); i++ ) {
				double[] a = samples.get(i);
								
				DoubleMatrix XtW = new DoubleMatrix(X.getColumns(),X.getRows());		
				for( int j = 0; j < X.getRows(); j++ ) {
					double[] b = samples.get(j);
					double d = gDist.dist( a, b );											
					double w = GeoUtils.getKernelValue(k, d, bandwidth.get(a));
					
					XtW.putColumn(j, X.getRow(j).mul(w));
				}	
				
				DoubleMatrix XtWX = XtW.mmul(X);				
				DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
				DoubleMatrix rowI = X.getRow(i);
				
				betas.putRow(i, beta);
				responseTrain.add( rowI.mmul(beta).get(0) );
				S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );		
			}
			
			double rss = SupervisedUtils.getResidualSumOfSquares(responseTrain, samples, ta);
			double mse = rss/samples.size();
			double traceS = S.diag().sum();
			
			log.debug("kernel: "+k);
			log.debug("bandwidth: "+bw);
			log.debug("Nr params: "+traceS);		
			log.debug("Nr. of data points: "+samples.size());
			/*double traceStS = S.transpose().mmul(S).diag().sum();
			log.debug("Effective nr. Params: "+(2*traceS-traceStS));
			log.debug("Effective degrees of freedom: "+(samplesTrain.size()-2*traceS+traceStS));*/
			log.debug("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size())); 
			log.debug("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, traceS, samples.size())); 
			log.debug("RSS: "+rss); 
			log.debug("R2: "+SupervisedUtils.getR2(responseTrain, samples, ta));			
		}
	}
}
