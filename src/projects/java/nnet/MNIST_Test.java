package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.Sigmoid;
import nnet.activation.SoftMax;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class MNIST_Test {
	
	private static Logger log = Logger.getLogger(MNIST_Test.class);

	public static void main(String[] args) {
		Random r = new Random();

		
		List<double[]> samplesTrain = new ArrayList<>();
		List<double[]> desiredTrain = new ArrayList<>();
		for (double[] d : DataUtils.readDataFrameFromCSV(new File("data/mnist_train.csv"), new int[] {}, true).samples) {
			samplesTrain.add(Arrays.copyOfRange(d, 1,d.length));
			double[] nd = new double[10];
			nd[(int)d[0]] = 1;
			
			desiredTrain.add( nd );
		}
		
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();
		for (double[] d : DataUtils.readDataFrameFromCSV(new File("data/mnist_test.csv"), new int[] {}, true).samples) {
			samplesVal.add(Arrays.copyOfRange(d, 1,d.length));
			double[] nd = new double[10];
			nd[(int)d[0]] = 1;
			
			desiredVal.add( nd );
		}
		
		List<double[]> allSamples = new ArrayList<>(samplesTrain);
		allSamples.addAll(samplesVal);
		DataUtils.transform(allSamples, Transform.zScore);
		
		List<Integer> nan = new ArrayList<>();
		for( int i = 0; i < samplesTrain.get(0).length; i++ )
			if( !nan.contains(i) && Double.isNaN(samplesTrain.get(0)[i]) || Double.isNaN(samplesVal.get(0)[i]) )
					nan.add(i);
		int[] toRemove = new int[nan.size()];
		for( int i = 0; i < toRemove.length; i++ )
			toRemove[i] = nan.get(i);
		log.debug("NaN: "+Arrays.toString(toRemove));
		
		samplesTrain = DataUtils.removeColumns(samplesTrain,toRemove);
		samplesVal = DataUtils.removeColumns(samplesVal,toRemove);
				
		for( double lr : new double[]{ 0.001 } ) {
					
			List<Function> input = new ArrayList<Function>();
			for( int i = 0; i < samplesVal.get(0).length; i++ )
				input.add( new Identity() );
			input.add( new Constant(1.0));
			
			List<Function> hidden = new ArrayList<Function>();
			for( int i = 0; i < 48; i++ )
				hidden.add( new Sigmoid() );
			hidden.add( new Constant(1.0));
			
			List<Function> output = new ArrayList<Function>();
			for( int i = 0; i < desiredTrain.get(0).length; i++ )
				output.add( new SoftMax() );
			//output.add( new Identity());
						
			log.debug(input.size());
			
			NNet nnet = new NNet( new Function[][]{
				input.toArray(new Function[]{}),
				hidden.toArray( new Function[]{}),
				output.toArray( new Function[]{} )
				}, lr );
			
			for (int i = 0; i < 500000; i++) {
				int idx = r.nextInt(samplesTrain.size());
				//nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
			}
			
			int incorrect = 0;
			for (int i = 0; i < samplesVal.size(); i++ ) {
				double[] a = nnet.present(samplesVal.get(i));
				
				if( i < 25 ) {
					log.debug(Arrays.toString(a));
					log.debug(Arrays.toString(desiredVal.get(i)));
					log.debug(max(a)+","+ max( desiredVal.get(i) ));
				}
												
				if( max(a) != max( desiredVal.get(i) ) )
					incorrect++;
			}
			log.debug("error rate: "+(double)incorrect/samplesVal.size());
		}
	}
	
	private static int max( double[] a ) {
		double m = Double.NEGATIVE_INFINITY;
		int j = -1;
		for( int i = 0; i < a.length; i++ ) {
			if( j < 0 || a[i] > m ) {
				m = a[i];
				j = i;
			}
		}
		return j;
	}
}
