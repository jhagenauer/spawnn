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
import nnet.activation.SoftMax;
import nnet.activation.TanH;
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
			//nd = new double[]{d[0]};
			//nd = Arrays.copyOf(nd,nd.length-1);
			desiredTrain.add( nd );
		}
		
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();
		for (double[] d : DataUtils.readDataFrameFromCSV(new File("data/mnist_test.csv"), new int[] {}, true).samples) {
			samplesVal.add(Arrays.copyOfRange(d, 1,d.length));
			double[] nd = new double[10];
			nd[(int)d[0]] = 1;
			//nd = new double[]{d[0]};
			//nd = Arrays.copyOf(nd,nd.length-1);
			desiredVal.add( nd );
		}
		
		List<double[]> allSamples = new ArrayList<>(samplesTrain);
		allSamples.addAll(samplesVal);
		DataUtils.transform(allSamples, Transform.scale01); // why does it give problems with zScore but not 01Score?
		
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
				
		for( double lr : new double[]{ 0.1 } ) {
					
			List<Function> input = new ArrayList<Function>();
			for( int i = 0; i < samplesVal.get(0).length; i++ )
				input.add( new Identity() );
			input.add( new Constant(1.0));
			
			List<Function> hidden1 = new ArrayList<Function>();
			for( int i = 0; i < 256; i++ )
				hidden1.add( new TanH() );
			hidden1.add( new Constant(1.0));
						
			List<Function> output = new ArrayList<Function>();
			for( int i = 0; i < desiredTrain.get(0).length; i++ ) {
				output.add( new SoftMax() );
				//output.add( new Identity());
			}
						
			NNet nnet = new NNet( new Function[][]{
				input.toArray(new Function[]{}),
				hidden1.toArray( new Function[]{}),
				output.toArray( new Function[]{} )
				}, lr );
			
			double lastE = Double.POSITIVE_INFINITY;
			for (int i = 0;; i++) {
				int idx = r.nextInt(samplesTrain.size());
				nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
				
				if( i % 100000 == 0 ) {
					int incorrect = 0;
					for (int j = 0; j < samplesVal.size(); j++ ) {
						double[] a = nnet.present(samplesVal.get(j));												
						if( max(a) != max( desiredVal.get(j) ) )
							incorrect++;
					}
					double e = (double)incorrect/samplesVal.size();
					
					log.debug(e+"\t"+i);
					if( e > lastE )
						break;
					lastE = e;
				}
			}
			
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
