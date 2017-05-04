package inc_llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.ConstantDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class PerformanceTest {

	private static Logger log = Logger.getLogger(PerformanceTest.class);

	public static void main(String[] args) {
		Random r = new Random(0);

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 52, 49, 10 };
		int ta = 7;
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);

		List<double[]> samplesTrain = new ArrayList<>();
		List<double[]> desiredTrain = new ArrayList<>();
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();

		for (double[] d : samples) {
			if (r.nextDouble() < 0.8) {
				samplesTrain.add(d);
				desiredTrain.add(new double[] { d[ta] });
			} else {
				samplesVal.add(d);
				desiredVal.add(new double[] { d[ta] });
			}
		}

		boolean gaussian = true;
		boolean adaptive = true;

		int t_max = 1000000;

		int aMax = 100;
		int lambda = 1000;
		double alpha = 0.5;
		double beta = 0.000005;

		Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);

		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < 2; i++) {
			double[] d = samples.get(r.nextInt(samples.size()));
			neurons.add(Arrays.copyOf(d, d.length));
		}

		long time = System.currentTimeMillis();
		for( int j = 0; j < 10; j++ ) {
			IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.005), 
					new ConstantDecay(0.005), 
					new PowerDecay(0.1, 0.000001),  
					new PowerDecay(0.1, 0.000001), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);
			for (int t = 0; t < t_max; t++) {
				int idx = r.nextInt(samplesTrain.size());
				llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));
			}
	
			List<double[]> responseVal = new ArrayList<double[]>();
			for (int i = 0; i < samplesVal.size(); i++)
				responseVal.add(llm.present(samplesVal.get(i)));
	
			log.debug("incLLM RMSE: " + SupervisedUtils.getRMSE(responseVal, desiredVal));
		}
		log.debug("took: "+(System.currentTimeMillis()-time)/1000);
	}
}
