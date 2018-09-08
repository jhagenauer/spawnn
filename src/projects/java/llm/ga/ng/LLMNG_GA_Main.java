package llm.ga.ng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import cern.colt.Arrays;
import ga.GeneticAlgorithm;
import heuristics.CostCalculator;
import llm.LLMNG;

public class LLMNG_GA_Main {

	private static Logger log = Logger.getLogger(LLMNG_GA_Main.class);

	public static void main(String[] args) {
		
		Random r = new Random(0);
		List<double[]> samples = new ArrayList<>();
		Map<Integer,Set<double[]>> cl = new HashMap<>();
		for( int i = 0; i < 1000; i++ ) {
			int c;
			double x1 = r.nextDouble();
			double x2 = r.nextDouble();
			double beta = 1;
			
			// data set 1: cluster defined by attributes as well as relationships (0,1 and 2 different relationship; 0,2 and 1 different attributes)
			/*if( x1 < 1.0/3 ) {
				c = 0;
			} else if ( x1 < 2.0/3 ) {
				x2 += 1;
				c = 1;
			} else {
				beta = -1;
				c = 2;
			}*/
			
			// data set 2: cluster defined by attributes
			if( x1 < 1.0/3 ) {
				c = 0;
			} else if ( x1 < 2.0/3 ) {
				x2 += 1;
				c = 1;
			} else {
				x2 -= 1;
				c = 2;
			}
									
			double[] d = new double[]{ x1, x2, beta, beta * x2};
			samples.add( d );
			
			if( !cl.containsKey(c) )
				cl.put(c, new HashSet<double[]>() );
			cl.get(c).add(d);			
		}
		
		int[] fa = new int[]{1}; // 1 == x2
		int[] ga = null;
		int ta = 3;
		//DataUtils.transform(samples, fa, Transform.zScore);
		
		CostCalculator<LLMNG_Individual> cc = new LLMNG_NMI_CostCalculator(samples,cl, fa,ta);
		//CostCalculator<LLM_Individual> cc = new LLM_CV_CostCalculator(samples, cl, fa, ta);
		
		for( LLMNG_Individual i : new LLMNG_Individual[]{
				
			// best for ds1:
			new LLMNG_Individual("{lr1Final=1.0E-5, lr1Func=Power, lr1Init=0.9, lr2Final=0.1, lr2Func=Power, lr2Init=0.9, mode=martinetz, nb1Final=0.1, nb1Func=Power, nb1Init=3.0, nb2Final=0.1, nb2Func=Power, nb2Init=3.0, t_max=100000, w=0.5}"),
						
			new LLMNG_Individual("{lr1Final=1.0E-5, lr1Func=Power, lr1Init=0.9, lr2Final=0.1, lr2Func=Power, lr2Init=0.9, mode=martinetz, nb1Final=0.1, nb1Func=Power, nb1Init=3.0, nb2Final=0.1, nb2Func=Power, nb2Init=3.0, t_max=100000, w=0.0}"),
			new LLMNG_Individual("{lr1Final=1.0E-5, lr1Func=Power, lr1Init=0.9, lr2Final=0.1, lr2Func=Power, lr2Init=0.9, mode=martinetz, nb1Final=0.1, nb1Func=Power, nb1Init=3.0, nb2Final=0.1, nb2Func=Power, nb2Init=3.0, t_max=100000, w=1.0}"),
		} ) {
			log.debug(i);
			log.debug( cc.getCost(i) );
			
			LLMNG llmng = i.train(samples, fa, ga, ta);
			for( double[] n : llmng.getNeurons() )
				log.debug("n: "+Arrays.toString(n)+", m: "+Arrays.toString( llmng.matrix.get(n)[0] )+", o: "+llmng.output.get(n)[0] );
		}
		//System.exit(1);
		 
		GeneticAlgorithm.tournamentSize = 3;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		
		List<LLMNG_Individual> init = new ArrayList<>();
		while (init.size() < 100) {
			init.add(new LLMNG_Individual());
		}
		
		GeneticAlgorithm<LLMNG_Individual> gen = new GeneticAlgorithm<LLMNG_Individual>();
		LLMNG_Individual result = (LLMNG_Individual) gen.search(init, cc);

		log.info("best:");
		log.info(cc.getCost(result));
		log.info(result.iParam);
	}
}
