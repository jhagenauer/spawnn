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
		DataUtils.transform(df.samples, Transform.zScore);
		
		List<double[]> samples = new ArrayList<>();
		List<double[]> desired = new ArrayList<>();
		for (double[] d : df.samples) {
			samples.add(Arrays.copyOf(d, 5));
			desired.add(new double[] { d[5] });
		}
		
		// rmse 0.29
		
		DescriptiveStatistics dsRMSE = new DescriptiveStatistics(), dsR2 = new DescriptiveStatistics();
		List<Entry<List<Integer>,List<Integer>>> cvList = SupervisedUtils.getCVList(5, 5, df.samples.size());
		for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList ) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(k));
				desiredTrain.add(desired.get(k));
			}
			
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) {
				samplesVal.add(samples.get(k));
				desiredVal.add(desired.get(k));
			}
			
			List<Function> input = new ArrayList<Function>();
			for( int i = 0; i < samplesVal.get(0).length; i++ )
				input.add( new Identity() );
			input.add( new Constant(1.0));
			
			List<Function> hidden = new ArrayList<Function>();
			for( int i = 0; i < 24; i++ )
				hidden.add( new TanH() );
			hidden.add( new Constant(1.0));
			
			NNet nnet = new NNet( new Function[][]{
				input.toArray(new Function[]{}),
				hidden.toArray( new Function[]{}),
				new Function[]{new Identity()}
				}, 0.01 );
			
			for (int i = 0; i < 100000; i++) {
				int idx = r.nextInt(samplesTrain.size());
				nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
			}
			
			List<double[]> response = new ArrayList<double[]>();
			for (double[] x : samplesVal)
				response.add(nnet.present(x));
			dsRMSE.addValue( SupervisedUtils.getRMSE(response, desiredVal) );
			dsR2.addValue( SupervisedUtils.getR2(response, desiredVal) );	
		}
		log.debug("RMSE: "+dsRMSE.getMean());
		log.debug("R2: "+dsR2.getMean());
	}

}
