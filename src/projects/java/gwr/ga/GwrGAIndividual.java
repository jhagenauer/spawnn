package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.utils.ColorBrewer;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;

public class GwrGAIndividual extends GAIndividual {

	protected List<Integer> bw;
	DoubleMatrix X,Y;
	List<double[]> samples;
	SpatialDataFrame sdf;
	Dist<double[]> gDist;
	
	Random r = new Random();

	public GwrGAIndividual( DoubleMatrix X, DoubleMatrix Y, List<Integer> bw, SpatialDataFrame sdf, Dist<double[]> gDist ) {
		this.X = X;
		this.Y = Y;
		this.bw = bw;
		this.samples = sdf.samples;
		this.sdf = sdf;
		this.gDist = gDist;
		
		if( X.getRows() != bw.size() )
			throw new RuntimeException("");
		this.cost = getCost();
	}
	
	@Override
	public GAIndividual mutate() {
		List<Integer> nBw = new ArrayList<>();
		for( int i = 0; i < samples.size(); i++ ) {
			int h = bw.get(i);
			
			if( r.nextDouble() < 1.0/samples.size() ) {
				/*if( h == X.getColumns() || r.nextBoolean() )
					h++;
				else
					h--;*/
				h = new int[]{8,9,10}[r.nextInt(3)];
			}
			nBw.add(h);
		}
		return new GwrGAIndividual(X, Y, nBw, sdf, gDist);
	}
	
	public List<Integer> getBandwidth() {
		return bw;
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {
		List<Integer> mBw = ((GwrGAIndividual)mother).getBandwidth();
		List<Integer> nBw = new ArrayList<>();
		for( int i = 0; i < bw.size(); i++)
			if( r.nextBoolean() )
				nBw.add(mBw.get(i));
			else
				nBw.add(bw.get(i));
		return new GwrGAIndividual(X, Y, nBw, sdf, gDist);
	}
	
	boolean gaussian = true;
	double cost = Double.NaN;
	
	@Override
	public double getCost() {
		if( !Double.isNaN(cost) )
			return cost;
		
		List<Double> residuals = new ArrayList<Double>();
		DoubleMatrix betas = new DoubleMatrix(X.getRows(),X.getColumns() );
		DoubleMatrix S = new DoubleMatrix( X.getRows(), X.getRows() );
		
		Map<double[],Double> bandwidth = new HashMap<>();
		for( int i = 0; i < samples.size(); i++ ) {
			double[] a = samples.get(i);
			int k = bw.get(i);
			List<double[]> s = new ArrayList<>(samples); 
			Collections.sort(s, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) { 
					return Double.compare( gDist.dist(o1, a), gDist.dist(o2, a)); 
				}
			});
			bandwidth.put(a, gDist.dist( s.get( k-1 ), a) );
						
		}
					
		for( int i = 0; i < samples.size(); i++ ) {
			double[] a = samples.get(i);
							
			DoubleMatrix XtW = new DoubleMatrix(X.getColumns(),X.getRows());		
			for( int j = 0; j < X.getRows(); j++ ) {
				double[] b = samples.get(j);
				double d = gDist.dist( a, b);
										
				double w;
				if( gaussian ) // Gaussian
					w = Math.exp(-0.5*Math.pow(d/bandwidth.get(a),2));
				else // bisquare
					w = Math.pow(1.0-Math.pow(d/bandwidth.get(a), 2), 2);
				XtW.putColumn(j, X.getRow(j).mul(w));
			}	
			
			DoubleMatrix XtWX = XtW.mmul(X);				
			DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
			DoubleMatrix rowI = X.getRow(i);
			
			betas.putRow(i, beta);
			residuals.add( Y.getRow(i).get(0) - rowI.mmul(beta).get(0) );
			S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );		
		}
		
		double rss = 0;
		for( double r : residuals )
			rss += r*r;
		double mse = rss/samples.size();
		double traceS = S.diag().sum();
		

		
		this.cost = SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size()); 	
		
		if( Double.isNaN(cost) ) {
			System.out.println(cost);
			System.out.println(rss+","+mse+","+traceS);
			System.exit(1);
		}
		return cost;
	}
	
	public void write(String fa, String fb) {
		List<double[]> values = new ArrayList<>();
		for( int i : bw )
			values.add( new double[]{i} );
		Drawer.geoDrawValues(sdf.geoms, values, 0, sdf.crs, ColorBrewer.Blues, ColorClass.Quantile, fa);
		
		DataUtils.writeShape(values, sdf.geoms, new String[]{"bandwidth"}, sdf.crs, fb);
	}
}
