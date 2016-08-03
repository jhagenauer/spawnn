package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.TanH;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class AirfoilTest {

	private static Logger log = Logger.getLogger(AirfoilTest.class);

	public static void main(String[] args) {
		Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/airfoil_self_noise.csv"), new int[] {}, true);
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		DataUtils.transform(df.samples, Transform.scale01);

		List<double[]> samples = new ArrayList<>();
		List<double[]> desired = new ArrayList<>();
		for (double[] d : df.samples) {
			samples.add(Arrays.copyOf(d, 5));
			desired.add(new double[] { d[5] });
		}
		
		// 24, 0.05		0.1247
		// 48, 0.01		0.1268
		// 24x24, 0.01	0.1196
		

		for (double lr : new double[] { 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 }) {
			log.debug("lr: "+lr);
			DescriptiveStatistics dsRMSE = new DescriptiveStatistics();
			List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 5, df.samples.size());
			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
				List<double[]> samplesTrain = new ArrayList<double[]>();
				List<double[]> desiredTrain = new ArrayList<double[]>();
				for (int k : cvEntry.getKey()) {
					samplesTrain.add(samples.get(k));
					desiredTrain.add(desired.get(k));
				}

				List<double[]> samplesVal = new ArrayList<double[]>();
				List<double[]> desiredVal = new ArrayList<double[]>();
				for (int k : cvEntry.getValue()) {
					samplesVal.add(samples.get(k));
					desiredVal.add(desired.get(k));
				}

				List<Function> input = new ArrayList<Function>();
				for (int i = 0; i < samplesVal.get(0).length; i++)
					input.add(new Identity());
				input.add(new Constant(1.0));

				List<Function> hidden1 = new ArrayList<Function>();
				for (int i = 0; i < 24; i++)
					hidden1.add(new TanH());
				hidden1.add(new Constant(1.0));
				
				List<Function> hidden2 = new ArrayList<Function>();
				for (int i = 0; i < 24; i++)
					hidden2.add(new TanH());
				hidden2.add(new Constant(1.0));

				NNet nnet = new NNet(new Function[][] { 
					input.toArray(new Function[] {}), 
					hidden1.toArray(new Function[] {}), 
					hidden2.toArray(new Function[] {}), 
					new Function[] { new Identity() } }, 
						lr );

				double lastE = Double.POSITIVE_INFINITY;
				for (int i = 0;; i++) {
					int idx = r.nextInt(samplesTrain.size());
					nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));

					if (i % 10000 == 0) {
						List<double[]> responseVal = new ArrayList<double[]>();
						for (double[] x : samplesVal)
							responseVal.add(nnet.present(x));
						double e = SupervisedUtils.getRMSE(responseVal, desiredVal);
						if (lastE < e || Double.isNaN(e))
							break;
						lastE = e;
					}
					
				}
				dsRMSE.addValue(lastE);
			}
			log.debug("RMSE: " + dsRMSE.getMean());
		}
	}

}
