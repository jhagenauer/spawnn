import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.jfree.util.Log;

import chowClustering.ChowClustering;
import chowClustering.ChowClustering.MyOLS;
import chowClustering.ChowClustering.StructChangeTestMode;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class TestStructChange {
	public static void main(String[] args) {

		{
			double[][] x1 = new double[][] { new double[] { 1, 0.5 }, new double[] { 1, 1.0 }, new double[] { 1, 1.5 }, new double[] { 1, 2.0 }, new double[] { 1, 2.5 }, new double[] { 1, 3.0 }, new double[] { 1, 3.5 } };
			double[][] x2 = new double[][] { new double[] { 1, 4.0 }, new double[] { 1, 4.5 }, new double[] { 1, 5.0 }, new double[] { 1, 5.5 }, new double[] { 1, 6.0 } };

			/*double[] y1 = new double[] { -0.043, 0.435, 0.149, 0.252, 0.571, 0.555, 0.678 };
			double[] y2 = new double[] { 3.119, 2.715, 3.671, 3.928, 3.962 };*/
			
			double[] y1 = new double[] { 2,3,4,5,6,7,8 };
			double[] y2 = new double[] { 9,10,11,12,13 };

			System.out.println("Chow: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.Chow)));
			System.out.println("ResiChow: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.ResiChow)));
			System.out.println("AdjustedChow: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.AdjustedChow)));
			System.out.println("Wald: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.Wald)));

			System.out.println("ResiSimple: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.ResiSimple)));
			System.out.println("ResiLikelihoodRatio: " + Arrays.toString(ChowClustering.testStructChange(x1, y1, x2, y2, StructChangeTestMode.LogLikelihood)));
		}
	}
}
