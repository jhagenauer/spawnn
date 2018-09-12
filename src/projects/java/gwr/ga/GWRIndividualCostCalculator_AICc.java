package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import nnet.SupervisedUtils;
import regioClust.LinearModel;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRIndividualCostCalculator_AICc extends GWRCostCalculator {
	
	public static boolean debug = false;
	private static Logger log = Logger.getLogger(GWRIndividualCostCalculator_AICc.class);
	
	GWRIndividualCostCalculator_AICc(List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, boolean adaptive) {
		super(samples, fa, ga, ta, kernel,adaptive);
	}

	@Override
	public double getCost(GWRIndividual_fixed ind) {
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
				log.warn("Couldn't solve eqs! Too low bandwidth?! real bw: "+bw+", adapt: "+adaptive+", gene: "+ind.getChromosome().get(i) );
				return Double.MAX_VALUE;
			}
			
			DoubleMatrix rowI = X.getRow(i);
			S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );	
		}
		double rss = SupervisedUtils.getResidualSumOfSquares(predictions, samples, ta);
		double mse = rss/samples.size();
		double traceS = S.diag().sum();	

		if( debug ) {
			log.debug("kernel: "+kernel);
			log.debug("Nr params: "+traceS);		
			log.debug("Nr. of data points: "+samples.size());
			double traceStS = S.transpose().mmul(S).diag().sum();
			log.debug("Effective nr. Params: "+(2*traceS-traceStS));
			log.debug("Effective degrees of freedom: "+(samples.size()-2*traceS+traceStS));
			log.debug("AICc: "+SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size())); 
			log.debug("AIC: "+SupervisedUtils.getAIC_GWMODEL(mse, traceS, samples.size())); 
			log.debug("RSS: "+rss); 
			log.debug("R2: "+SupervisedUtils.getR2(predictions, samples, ta));
		}
		double aic = SupervisedUtils.getAICc_GWMODEL(mse, traceS, samples.size());
		if( debug )
			log.debug("AIC: "+aic);
		
		if( Double.isInfinite(mse) || Double.isNaN(mse) )
			throw new RuntimeException("aic "+aic+" mse "+mse);
		
		return aic;
	}
}
