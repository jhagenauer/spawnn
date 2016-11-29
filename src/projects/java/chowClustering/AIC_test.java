package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import chowClustering.ChowClustering.MyOLS;
import nnet.SupervisedUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class AIC_test {

	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/chicago.shp"), true);
		int[] fa = new int[]{3,7,9,10};
		int ta = 6;
		
		double[][] x = ChowClustering.getX( sdf.samples, fa, true);
		double[] y = ChowClustering.getY( sdf.samples, ta);
				
		OLSMultipleLinearRegression ols = new MyOLS();
		ols.setNoIntercept(true);
		ols.newSampleData(y, x);
		double[] beta = ols.estimateRegressionParameters();
		
		List<Double> response = new ArrayList<Double>();
		for (int i = 0; i < x.length; i++) {
			double p = 0;
			for (int j = 0; j < beta.length; j++)
				p += beta[j] * x[i][j];
			response.add(p);					
		}
		
		double rss = SupervisedUtils.getResidualSumOfSquares(response, sdf.samples, ta);
						
		System.out.println("rss: "+rss);
		
		int nrParams = beta.length;
		int nrSamples = sdf.samples.size();	
		System.out.println("AIC:  "+SupervisedUtils.getAIC_GWMODEL(rss/nrSamples, nrParams, nrSamples ) );
		System.out.println("AICc: "+SupervisedUtils.getAICc_GWMODEL(rss/nrSamples, nrParams, nrSamples ) );
		
		System.out.println("pre: "+nrSamples * ( Math.log(rss/nrSamples) + Math.log(2*Math.PI) + 1 ));
		System.out.println("a: "+(2 * (nrParams+1)));
		System.out.println("b: "+((2 * nrSamples * ( nrParams + 1 ) ) / (nrSamples - nrParams - 2)));
		System.out.println("c: "+((2 * nrParams * (nrParams + 1)) / (nrSamples - nrParams - 1)));
	}
}
