package rbf.twoplanes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.IncRBF;
import spawnn.rbf.RBF;
import spawnn.utils.DataUtils;
import cern.colt.Arrays;

public class TwoPlanes_NoCV_FixedNeurons {

	private static Logger log = Logger.getLogger(TwoPlanes_NoCV_FixedNeurons.class);

	public static void main(String[] args) {
		
		List<double[]> all = DataUtils.readCSV("data/twoplanes.csv");

		final Dist<double[]> dist = new EuclideanDist();

		Collections.shuffle(all);
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : all) {
			samples.add(new double[] { d[0], d[1] });
			desired.add(new double[] { d[2] });
		}
		
		List<double[]> responseA = new ArrayList<double[]>();
		{
			// old
			/*double[] n1 =  new double[]{ -0.5166669157051942,0.4747797150178334 };
			double r1 = 0.9851985826882976;
			double w1 = -2.3752056274424205;
			
			double[] n2 = new double[]{ 0.4659087442445063,0.4750006164127265 };
			double r2 = r1;
			double w2 = 2.374174404890089;*/
			
			double[] n1 =  new double[]{ -0.5001859706,	0.4748332352 };
			double r1 = 0.9520545825;
			double w1 = -2.3494862729;
			
			double[] n2 = new double[]{ 0.4508517147, 0.4752162914 };
			double r2 = r1;
			double w2 = 2.349620619;
														
			for (double[] x : samples) {
				double r = w1 * Math.exp( -0.5 * Math.pow(dist.dist(x, n1) / r1, 2) );
				r += w2 * Math.exp( -0.5 * Math.pow(dist.dist(x, n2) / r2, 2) );
				responseA.add( new double[]{ r } );
			}
			
			double[] cm = TwoPlanes.getConfusionMatrix(responseA, desired);
			double precision = cm[0] / (cm[0] + cm[2]); // tp durch alle p
			double recall = cm[0] / (cm[0] + cm[1]); // tp durch all true
			double fmeasure = 2 * (precision * recall) / (precision + recall);
	
			log.debug("A:");
			log.debug("cm: "+Arrays.toString(cm));
			log.debug("f-score: " + fmeasure);
			log.debug("RMSE: " + Meuse.getRMSE(responseA, desired));
			log.debug("R^2: " + Math.pow(Meuse.getPearson(responseA, desired), 2));
		}
		
		List<double[]> responseB = new ArrayList<double[]>();
		{
			// old
			/*double[] n1 = new double[]{ -0.5158912087744072,0.47510917579418227};
			double r1 = 0.5827596538833388;
			double w1 = -1.6170130128230986;
			
			double[] n2 = new double[]{0.4663431642172851,0.4748200707442413};
			double r2 = r1;
			double w2 = 1.6163284849649429;*/
			
			double[] n1 =  new double[]{ -0.5006867257,	0.4751496116 };
			double r1 = 0.5833419363;
			double w1 = -1.7073244544;
			
			double[] n2 = new double[]{ 0.4506347709, 0.4747955239 };
			double r2 = r1;
			double w2 = 1.7065796928;
				
			for (double[] x : samples) {
				double r = w1 * Math.exp( -0.5 * Math.pow(dist.dist(x, n1) / r1, 2) );
				r += w2 * Math.exp( -0.5 * Math.pow(dist.dist(x, n2) / r2, 2) );
				responseB.add( new double[]{ r } );
			}
			
			double[] cm = TwoPlanes.getConfusionMatrix(responseB, desired);
			double precision = cm[0] / (cm[0] + cm[2]); // tp durch alle p
			double recall = cm[0] / (cm[0] + cm[1]); // tp durch all true
			double fmeasure = 2 * (precision * recall) / (precision + recall);
	
			log.debug("B:");
			log.debug("cm: "+Arrays.toString(cm));
			log.debug("f-score: " + fmeasure);
			log.debug("RMSE: " + Meuse.getRMSE(responseB, desired));
			log.debug("R^2: " + Math.pow(Meuse.getPearson(responseB, desired), 2));
		}
		
		// write
		List<double[]> out = new ArrayList<double[]>();
		for( int i = 0; i < samples.size(); i++ )
			out.add( TwoPlanes.append( samples.get(i), desired.get(i), responseA.get(i), responseB.get(i)));
		DataUtils.writeCSV("output/response.csv", out, new String[]{"x","y","z","predictionA","predictionB"} );
	}
}
