package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ColorBrewer;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.DataUtils;

public class GwrGAIndividual extends GAIndividual {

	protected List<Integer> bw;
	List<double[]> samples;
	SpatialDataFrame sdf;
	int[] i, fa, ga;
	int ta;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	Random r = new Random();

	// TODO: move cost-related parameters to one class/function... this sucks here. Question is where to use costcalculator?
	public GwrGAIndividual( int[] i, List<Integer> bw, SpatialDataFrame sdf, List<Entry<List<Integer>, List<Integer>>> cvList, int[] fa, int[] ga, int ta ) {
		this.bw = bw;
		this.samples = sdf.samples;
		this.sdf = sdf;
		this.cvList = cvList;
		this.i = i;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		
		this.cost = getCost();
	}
	
	@Override
	public GAIndividual mutate() {
		List<Integer> nBw = new ArrayList<>();
		for( int j = 0; j < samples.size(); j++ ) {
			int h = bw.get(j);
			
			if( r.nextDouble() < 1.0/samples.size() ) {
				/*if( h == X.getColumns() || r.nextBoolean() )
					h++;
				else
					h--;*/
				h = i[r.nextInt(i.length)];
			}
			nBw.add(h);
		}
		return new GwrGAIndividual(i, nBw, sdf, cvList, fa, ga, ta );
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
		return new GwrGAIndividual(i, nBw, sdf, cvList , fa,ga, ta );
	}
	
	double cost = Double.NaN;
	
	@Override
	public double getCost() {
		if( !Double.isNaN(cost) )
			return cost;
				
		Dist<double[]> gDist = new EuclideanDist(ga);
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
		
		SummaryStatistics ss = new SummaryStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
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
					double w = GeoUtils.getKernelValue( GWKernel.gaussian, d, bandwidth.get(a));

					XtW.putColumn(j, X.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(X);
				DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));

				predictions.add(XVal.getRow(i).mmul(beta).get(0));
			}
			ss.addValue( SupervisedUtils.getRMSE(predictions, samplesVal, ta) );
		}			
		this.cost = ss.getMean();
		
		if( Double.isNaN(cost) ) {
			System.out.println(cost);
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
