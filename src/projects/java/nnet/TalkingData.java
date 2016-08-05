package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.math.IEEE754rUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.Sigmoid;
import nnet.activation.SoftMax;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class TalkingData {
	
	private static Logger log = Logger.getLogger(TalkingData.class);

	public static void main(String[] args) {
		Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/talkingData.csv"), new int[] {}, true);
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		
		List<double[]> samples = new ArrayList<>();
		List<double[]> desired = new ArrayList<>();
		for (double[] d : df.samples) {
			samples.add( Arrays.copyOfRange(d, 12+1, d.length-1));
			
			/*double sum = 0;
			for( double a : Arrays.copyOfRange(d, 0, 12))
				sum += a;
			desired.add( new double[]{sum} );*/
			desired.add(  Arrays.copyOfRange(d, 0, 12) );
		}
		
		DataUtils.transform(samples, Transform.scale01);
								
		DescriptiveStatistics ds = new DescriptiveStatistics();
		List<Entry<List<Integer>,List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, df.samples.size());
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
			for( int i = 0; i < 100; i++ )
				hidden.add( new Sigmoid() );
			hidden.add( new Constant(1.0));
			
			List<Function> output = new ArrayList<Function>();
			for( int i = 0; i < desiredTrain.get(0).length; i++ ) 
				output.add( new SoftMax() );
			//output.add( new Identity() );
					
			NNet nnet = new NNet( new Function[][]{
				input.toArray(new Function[]{}),
				hidden.toArray( new Function[]{}),
				output.toArray( new Function[]{} )
				}, 0.01 );
			
			for (int i = 0; i < 10000; i++) {
				int idx = r.nextInt(samplesTrain.size());
				nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
			}
			
			List<double[]> response = new ArrayList<double[]>();
			for (double[] x : samplesVal) {
				double[] re = nnet.present(x);
				log.debug(Arrays.toString(re));
				System.exit(1);
				response.add(re);
			}
			double ll = SupervisedUtils.getMultiLogLoss( response, desiredVal);
			log.debug(ll);
			ds.addValue( ll );	
		}
		log.debug("logLoss: "+ds.getMean());
	}
}
