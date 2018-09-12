package gwr.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import regioClust.LinearModel;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRIndividualCostCalculatorCorrelation<T extends GWRIndividual<T>> extends GWRCostCalculator<T> {

	public static boolean debug = false;
	private String[] faNames;
	private static Logger log = Logger.getLogger(GWRIndividualCostCalculatorCorrelation.class);

	public GWRIndividualCostCalculatorCorrelation(List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, String[] faNames) {
		super(samples, fa, ga, ta, kernel);
		List<String> l = new ArrayList<>(Arrays.asList(faNames));
		l.add(0, "Intercept");
		this.faNames = l.toArray(new String[] {});
	}

	@Override
	public double getCost(T ind) {
		Map<double[], Double> bandwidth = ind.getSpatialBandwidth(samples, new EuclideanDist(ga) );

		DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samples, ta));
		DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samples, fa, true));

		double[][] betas = new double[samples.size()][];
		DoubleMatrix S = new DoubleMatrix(X.getRows(), X.getRows());

		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			double bw = bandwidth.get(a);

			DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
			for (int j = 0; j < X.getRows(); j++) {
				double[] b = samples.get(j);
				double d = gDist.dist(a, b);
				double w = GeoUtils.getKernelValue(kernel, d, bw);

				XtW.putColumn(j, X.getRow(j).mul(w));
			}
			DoubleMatrix XtWX = XtW.mmul(X);

			try {
				DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
				betas[i] = beta.data;
			} catch (LapackException e) {
				log.warn("Couldn't solve eqs! Too low bandwidth?! real bw: "+bw+" , gene: "+ind.geneToString(i) );
				return Double.MAX_VALUE;
			}

			DoubleMatrix rowI = X.getRow(i);
			S.putRow(i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW));
		}

		PearsonsCorrelation pc = new PearsonsCorrelation(betas);

		RealMatrix cor = pc.getCorrelationMatrix();
		RealMatrix pValues = pc.getCorrelationPValues();

		SummaryStatistics ss = new SummaryStatistics();
		for (int i = 1; i < cor.getColumnDimension() - 1; i++) { // no intercept
			for (int j = i + 1; j < cor.getColumnDimension(); j++) {
				if (debug) {
					log.debug("cor " + faNames[i] + " " + faNames[j] + " : " + cor.getEntry(i, j) + ", " + pValues.getEntry(i, j));
				}
				if( pValues.getEntry(i,j) < 0.05 )
					ss.addValue( Math.abs( cor.getEntry(i, j) ) );
			}
		}
		return ss.getMean();
	}
}
