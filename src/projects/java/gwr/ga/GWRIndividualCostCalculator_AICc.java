package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWRIndividualCostCalculator_AICc implements CostCalculator<GWRIndividual> {

	SpatialDataFrame sdf;
	List<double[]> samples;
	int[] fa, ga;
	int ta;

	public GWRIndividualCostCalculator_AICc(SpatialDataFrame sdf, int[] fa, int[] ga, int ta) {
		this.samples = sdf.samples;
		this.sdf = sdf;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
	}

	@Override
	public double getCost(GWRIndividual ind) {
		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Double> bandwidth = new HashMap<>();
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			int k = ind.getBandwidthAt(i);
			List<double[]> s = new ArrayList<>(samples);
			Collections.sort(s, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
				}
			});
			bandwidth.put(a, gDist.dist(s.get(k - 1), a));
		}

		DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samples, ta));
		DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samples, fa, true));

		List<Double> predictions = new ArrayList<>();
		DoubleMatrix S = new DoubleMatrix( X.getRows(), X.getRows() );
		
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);

			DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
			for (int j = 0; j < X.getRows(); j++) {
				double[] b = samples.get(j);
				double d = gDist.dist(a, b);
				double w = GeoUtils.getKernelValue(GWKernel.gaussian, d, bandwidth.get(a));

				XtW.putColumn(j, X.getRow(j).mul(w));
			}
			DoubleMatrix XtWX = XtW.mmul(X);
			DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));

			predictions.add(X.getRow(i).mmul(beta).get(0));
		}
		double rss = SupervisedUtils.getResidualSumOfSquares(predictions, samples, ta);
		double mse = rss/samples.size();
		double traceS = S.diag().sum();		
		return SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size()); 
	}
}
