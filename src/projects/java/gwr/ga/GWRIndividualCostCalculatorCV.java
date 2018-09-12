package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import nnet.SupervisedUtils;
import regioClust.LinearModel;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRIndividualCostCalculatorCV<T extends GWRIndividual<T>> extends GWRCostCalculator<T> {
	
	public static boolean debug = false;
	private static Logger log = Logger.getLogger(GWRIndividualCostCalculatorCV.class);
		
	List<Entry<List<Integer>, List<Integer>>> cvList;
		
	public GWRIndividualCostCalculatorCV( List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, int folds ) {
		super(samples,fa,ga,ta,kernel);
		this.cvList = SupervisedUtils.getCVList(folds, 1, samples.size() );
	}

	@Override
	public double getCost(T ind) {	
		Map<double[], Double> bandwidth = ind.getSpatialBandwidth(samples, new EuclideanDist(ga) );
		
		SummaryStatistics ss = new SummaryStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			
			List<double[]> samplesTrain = new ArrayList<double[]>();
			for (int k : cvEntry.getKey())
				samplesTrain.add(samples.get(k));

			List<double[]> samplesVal = new ArrayList<double[]>();
			for (int k : cvEntry.getValue())
				samplesVal.add(samples.get(k));

			DoubleMatrix YTrain = new DoubleMatrix(LinearModel.getY(samplesTrain, ta));
			DoubleMatrix XTrain = new DoubleMatrix(LinearModel.getX(samplesTrain, fa, true));

			DoubleMatrix XVal = new DoubleMatrix(LinearModel.getX(samplesVal, fa, true));
			
			List<Double> predVal = new ArrayList<>();
			for (int i = 0; i < samplesVal.size(); i++) {
				double[] a = samplesVal.get(i);
				double bw = bandwidth.get(a);

				DoubleMatrix XtW = new DoubleMatrix(XTrain.getColumns(), XTrain.getRows());
				for (int j = 0; j < XTrain.getRows(); j++) {
					double[] b = samplesTrain.get(j);
					double w = GeoUtils.getKernelValue( kernel, gDist.dist(a, b), bw );

					XtW.putColumn(j, XTrain.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(XTrain);
								
				try {
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(YTrain));
					predVal.add(XVal.getRow(i).mmul(beta).get(0));
				} catch( LapackException e ) {
					int idx = samples.indexOf(a);
					log.warn("Couldn't solve eqs! Too low bandwidth?! real bw: "+bw+" , gene: "+ind.geneToString(i) );
					return Double.POSITIVE_INFINITY;
				}				
			}
			double rmse = SupervisedUtils.getRMSE(predVal, samplesVal, ta);
			
			if( debug )
				log.debug("RMSE fold "+cvList.indexOf(cvEntry)+ ": "+rmse);
			
			ss.addValue( rmse );
		}
		if( debug )
			log.debug( "Mean RMSE: "+ss.getMean() );
		return ss.getMean();
	}
}