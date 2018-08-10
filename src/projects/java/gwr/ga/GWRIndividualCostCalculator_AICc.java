package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import nnet.SupervisedUtils;
import regioClust.LinearModel;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRIndividualCostCalculator_AICc extends GWRCostCalculator {
	
	GWRIndividualCostCalculator_AICc(List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, boolean adaptive,double minBw) {
		super(samples, fa, ga, ta, kernel,adaptive,minBw);
	}

	@Override
	public double getCost(GWRIndividual ind) {
		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Double> bandwidth = getSpatialBandwidth(ind);

		DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samples, ta));
		DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samples, fa, true));

		List<Double> predictions = new ArrayList<>();
		DoubleMatrix S = new DoubleMatrix( X.getRows(), X.getRows() );
		
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			double bw = bandwidth.get(a);

			DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
			for (int j = 0; j < X.getRows(); j++) {
				double[] b = samples.get(j);
				double d = gDist.dist(a, b);
				double w = GeoUtils.getKernelValue(kernel, d, bw );

				XtW.putColumn(j, X.getRow(j).mul(w));
			}
			DoubleMatrix XtWX = XtW.mmul(X);
			
			try {
				DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
				predictions.add(X.getRow(i).mmul(beta).get(0));
			} catch( LapackException e ) {
				System.err.println("Couldn't solve eqs! Too low bandwidth?! "+bw+", "+adaptive+", "+ind.getChromosome().get(i) );
				return Double.MAX_VALUE;
			}
			
			DoubleMatrix rowI = X.getRow(i);
			S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );	
		}
		double rss = SupervisedUtils.getResidualSumOfSquares(predictions, samples, ta);
		double mse = rss/samples.size();
		double traceS = S.diag().sum();	
		
		boolean debug = false;
		if( debug ) {
			System.out.println("kernel: "+kernel);
			System.out.println("Nr params: "+traceS);		
			System.out.println("Nr. of data points: "+samples.size());
			double traceStS = S.transpose().mmul(S).diag().sum();
			System.out.println("Effective nr. Params: "+(2*traceS-traceStS));
			System.out.println("Effective degrees of freedom: "+(samples.size()-2*traceS+traceStS));
			System.out.println("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size())); 
			System.out.println("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, traceS, samples.size())); 
			System.out.println("RSS: "+rss); 
			System.out.println("R2: "+SupervisedUtils.getR2(predictions, samples, ta));
		}
		
		return SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size()); 
	}
}
