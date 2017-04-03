package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.Sigmoid;
import nnet.activation.SoftMax;
import nnet.activation.TanH;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;

public class Quora {

	private static Logger log = Logger.getLogger(Quora.class);

	public static void main(String[] args) {
		Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/quora.csv"), new int[] {}, true,';');
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		
		List<double[]> samples = df.samples;
				
		int[] fa = new int[samples.get(0).length-1];
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+1;
		int ta = 0;
		
		DataUtils.transform(samples, fa, Transform.zScore);
		//DataUtils.transform(samples, fa, Transform.scale01);
		
		List<double[]> ss = new ArrayList<>();
		for( double[] d : samples )
			if( !Double.isNaN(d[ta]) )
				ss.add(d);
		samples = ss;
						
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		
		for (double lr : new double[] { 0.005 }) {
			log.debug("lr: "+lr);
			DescriptiveStatistics dsRMSE = new DescriptiveStatistics();
						
			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
				List<double[]> samplesTrain = new ArrayList<double[]>();
				for (int k : cvEntry.getKey()) 
					samplesTrain.add(samples.get(k));
					
				List<double[]> samplesVal = new ArrayList<double[]>();
				for (int k : cvEntry.getValue()) 
					samplesVal.add(samples.get(k));
				
				List<Function> input = new ArrayList<Function>();
				for (int i = 0; i < fa.length; i++) 
					input.add(new Identity());
				input.add(new Constant(1.0));

				List<Function> hidden1 = new ArrayList<Function>();
				for (int i = 0; i < 12; i++)
					hidden1.add(new TanH());
				hidden1.add(new Constant(1.0));
								
				NNet nnet = new NNet(new Function[][] { 
					input.toArray(new Function[] {}), 
					hidden1.toArray(new Function[] {}),
					//hidden2.toArray(new Function[] {}),
					//new Function[] { new Identity() } }, // 0.6560209292200196
					//new Function[] { new SoftMax() } }, // 
					new Function[] { new Sigmoid() } }, // 
						lr );

				double bestE = Double.POSITIVE_INFINITY;
				int noImp = 0;
				for (int i = 0;; i++) {
					double[] d = samplesTrain.get( r.nextInt(samplesTrain.size()) );
					nnet.train(i, DataUtils.strip(d, fa), new double[]{ d[ta] } );
										
					if (i % 10000 == 0) {
						List<Double> responseVal = new ArrayList<>();
						for (double[] d2 : samplesVal) 
							responseVal.add(nnet.present(DataUtils.strip(d2, fa))[0]);
						double e = SupervisedUtils.getMultiLogLoss(responseVal, samplesVal, ta);
						
						if( Double.isNaN(e) || noImp == 10 )
							break;
						
						if ( e <= bestE ) {
							bestE = e;
							noImp = 0;
						} else 
							noImp++;
						
						log.debug(i+","+noImp+","+e);
					}
				}
				log.debug(bestE);
				dsRMSE.addValue(bestE);
			}
			log.debug("LogLoss: " + dsRMSE.getMean());
		}
	}
}
