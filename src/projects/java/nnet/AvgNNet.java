package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;

public class AvgNNet {

	private static Logger log = Logger.getLogger(AvgNNet.class);

	public static void main(String[] args) {
		Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/airfoil_self_noise.csv"), new int[] {}, true);
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		DataUtils.transform(df.samples, transform.zScore);

		List<double[]> samples = new ArrayList<>();
		List<double[]> desired = new ArrayList<>();
		for (double[] d : df.samples) {
			samples.add(Arrays.copyOf(d, 5));
			desired.add(new double[] { d[5] });
		}

		for (double noise : new double[] { 0.0, 0.05 }) {

			DescriptiveStatistics dsRMSE = new DescriptiveStatistics(), dsR2 = new DescriptiveStatistics();
			List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(5, 5, df.samples.size());
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

				NNet[] nnet = new NNet[10];
				for (int i = 0; i < nnet.length; i++)
					nnet[i] = new NNet(new int[] { samplesVal.get(0).length, 24, 1 }, true, 0.01 );

				for (int i = 0; i < 400000; i++) {
					int idx = r.nextInt(samplesTrain.size());
					for (NNet n : nnet)
						n.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
				}

				List<double[]> response = new ArrayList<double[]>();
				for (double[] x : samplesVal) {
					List<double[]> re = new ArrayList<>();
					for (NNet n : nnet)
						re.add(n.present(x));
					response.add(DataUtils.getMean(re));
				}
				dsRMSE.addValue(SupervisedUtils.getRMSE(response, desiredVal));
				dsR2.addValue(SupervisedUtils.getR2(response, desiredVal));

			}
			log.debug("noise: " + noise);
			log.debug("RMSE: " + dsRMSE.getMean());
			log.debug("R2: " + dsR2.getMean());
		}
	}

}
