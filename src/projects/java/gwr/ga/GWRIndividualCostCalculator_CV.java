package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWRIndividualCostCalculator_CV implements CostCalculator<GWRIndividual> {
	
	SpatialDataFrame sdf;
	List<double[]> samples;
	int[] fa, ga;
	int ta;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	GWKernel kernel;
	
	public GWRIndividualCostCalculator_CV( SpatialDataFrame sdf, List<Entry<List<Integer>, List<Integer>>> cvList, int[] fa, int[] ga, int ta, GWKernel kernel ) {
		this.samples = sdf.samples;
		this.sdf = sdf;
		this.cvList = cvList;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		this.kernel = kernel;
	}

	@Override
	public double getCost(GWRIndividual ind) {		
		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[],Double> bandwidth = new HashMap<>();
		for( int i = 0; i < samples.size(); i++ ) {
			double[] a = samples.get(i);
			int k = ind.getBandwidthAt(i);
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
					double w = GeoUtils.getKernelValue( kernel, d, bandwidth.get(a));

					XtW.putColumn(j, X.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(X);
								
				try {
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
					predictions.add(XVal.getRow(i).mmul(beta).get(0));
				} catch( LapackException e ) {
					e.printStackTrace();
					System.err.println("!!!! "+ind.getBandwidthAt(i));
					return Double.MAX_VALUE;
				}				
			}
			ss.addValue( SupervisedUtils.getRMSE(predictions, samplesVal, ta) );
		}	
		return ss.getMean();
	}

}
