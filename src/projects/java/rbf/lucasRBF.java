package rbf;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import llm.LLMNG;
import llm.WeightedErrorSorter;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class lucasRBF {
	
	private static Logger log = Logger.getLogger(lucasRBF.class);

	enum sorter_mode {
		kangas, error
	};

	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/lucas/lucas.shp"), true);
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, sdf.size());

		int[] fa = new int[] { 2, // TLA
				3, // beds
				9, // rooms
				10, // lotsize
				19, // age
		};
		int[] ga = new int[] { 20, 21 };
		int ta = 0;
		
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		for (double[] d : sdf.samples) {
			d[19] = Math.pow(d[19], 2);
			d[10] = Math.log(d[10]);
			d[2] = Math.log(d[2]);
			d[1] = Math.log(d[1]);
		}
		
		int t_max = 100000;
		int numNeurons = 20;
		
		DecayFunction lr1 = new PowerDecay(0.9, 0.00005);
		DecayFunction lr2 = new PowerDecay(0.9, 0.1);
		DecayFunction nb = new PowerDecay(3, 0.1);
				
		Map<sorter_mode,SummaryStatistics> results = new HashMap<>();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			Random r1 = new Random(0);

			List<double[]> training = new ArrayList<double[]>();
			List<double[]> testing = new ArrayList<double[]>();
			for (int k : cvEntry.getKey()) 
				training.add( sdf.samples.get(k) );
			for (int k : cvEntry.getValue()) 
				testing.add( sdf.samples.get(k) );
								
			DataUtils.transform(training, Transform.zScore);
			DataUtils.transform(testing, Transform.zScore);

			for (sorter_mode m : sorter_mode.values()) {
				
				
				Map<double[], Set<double[]>> map = null;
				if (m == sorter_mode.kangas) {
					
					List<double[]> neurons = new ArrayList<>();
					while( neurons.size() < numNeurons ) {
						double[] x = training.get(r1.nextInt(training.size()));
						neurons.add( Arrays.copyOf(x, x.length ));
					}
											
					Sorter<double[]> sorter = new KangasSorter<double[]>(gDist, fDist, 1);
					NG ng = new NG(neurons, nb, lr1, sorter);
					for (int t = 0; t < t_max; t++) {
						double[] x = training.get(r1.nextInt(training.size()));
						ng.train((double) t / t_max, x);
					}
					
					map = NGUtils.getBmuMapping(training, ng.getNeurons(), sorter);					
				} else if ( m == sorter_mode.error ){
					
					WeightedErrorSorter wes =  new WeightedErrorSorter(null, fDist, training, ta, 0.1);
					
					List<double[]> neurons = new ArrayList<>();
					while (neurons.size() < numNeurons ) {
						double[] d = training.get(r1.nextInt(training.size()));
						neurons.add( Arrays.copyOf(d,d.length) );
					}					
					LLMNG llmng = new LLMNG(neurons, nb, lr1, nb, lr2, wes, fa, 1);
					llmng.aMode = false;
					wes.setSupervisedNet(llmng);
					
					for (int t = 0; t < t_max; t++) {
						double[] d = training.get(r1.nextInt( training.size() ));
						llmng.train((double) t / t_max, d, new double[] { d[ta] } );
					}
					map = NGUtils.getBmuMapping(training, llmng.getNeurons(), wes);					
				} 
				
				Map<double[], Double> hidden = new HashMap<double[], Double>();
				for (double[] n : map.keySet()) {						
					double sigma = 0;
					for (double[] x : map.get(n))
						sigma += Math.pow(gDist.dist(x, n), 2);
					sigma = Math.sqrt(sigma/map.get(n).size());						
					hidden.put(n, sigma);
				}

				RBF rbf = new RBF(hidden, 1, gDist, 0.001);
				for (int t = 0; t < t_max; t++) {
					double[] d = training.get(r1.nextInt( training.size() ));
					rbf.train(d, new double[]{ d[ta]} );
				}

				List<Double> response = new ArrayList<>();
				for (double[] x : testing)
					response.add(rbf.present(x)[0]);
				
				double r2 = SupervisedUtils.getR2(response, testing, ta);					
				if( !results.containsKey(m) )
					results.put(m, new SummaryStatistics() );
				results.get(m).addValue(r2);
			}
		}
		for( Entry<sorter_mode, SummaryStatistics> e : results.entrySet() ) 
			log.debug(e.getKey()+"\t"+e.getValue().getMean());
	}
}
