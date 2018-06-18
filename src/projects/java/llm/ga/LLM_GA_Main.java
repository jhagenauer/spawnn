package llm.ga;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import ga.GeneticAlgorithm;

public class LLM_GA_Main {

	private static Logger log = Logger.getLogger(LLM_GA_Main.class);

	public static void main(String[] args) {
			
			GeneticAlgorithm.tournamentSize = 2;	
			GeneticAlgorithm.elitist = true;
			GeneticAlgorithm.recombProb = 0.7;
			
			LLM_CostCalculator cc = new LLM_CostCalculator();
			List<LLM_Individual> init = new ArrayList<>();
			while (init.size() < 25) 
				init.add(new LLM_Individual() );
			
			GeneticAlgorithm<LLM_Individual> gen = new GeneticAlgorithm<LLM_Individual>();
			LLM_Individual result = (LLM_Individual) gen.search(init, cc);
			log.info("best:");
			log.info(cc.getCost(result));
			log.info(result.iParam);
	}
}
