package heuristics.tabu;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

public class TabuSearch<T extends TabuIndividual<T>> {
	
	private static Logger log = Logger.getLogger(TabuSearch.class);

	private Evaluator<T> eva;
	private int tlLength = 10, noImproMax = 250, noImprosUntilPenalty = 50, maxPenaltyDuration = 25;
	
	public static boolean rndMoveDiversication = false;
	
	public TabuSearch(Evaluator<T> evaluator, int tlLenght, int noImproMax ) {
		this.eva = evaluator;
		this.tlLength = tlLenght;
		this.noImproMax = noImproMax;
		// no penalty
		this.noImprosUntilPenalty = noImproMax;
		this.maxPenaltyDuration = 0;
	}
	
	public TabuSearch(Evaluator<T> evaluator, int tlLenght, int noImproMax, int noImprosUnitlPenalty, int maxPenaltyDuration ) {
		this.eva = evaluator;
		this.tlLength = tlLenght;
		this.noImproMax = noImproMax;
		this.noImprosUntilPenalty = noImprosUnitlPenalty;
		this.maxPenaltyDuration = maxPenaltyDuration;
	}

	public T search(T init) {
		TabuList<TabuMove<T>> tabuList = new TabuList<TabuMove<T>>(); // recency
		Map<Object,Integer> freq = new HashMap<Object,Integer>(); // frequency, long-term

		T curBest = init; // aktuell
		curBest.setValue(eva.evaluate(curBest));

		T globalBest = curBest; // global Best
		int noImpro = 0;	
		
		int k = 0;
		while( noImpro < noImproMax ) {
			k++;
			Set<TabuMove<T>> moves = new HashSet<>(curBest.getNeighboringMoves());
			
			if( rndMoveDiversication && noImpro > noImprosUntilPenalty && noImpro % noImprosUntilPenalty < maxPenaltyDuration ) {
				int s = moves.size();
				moves.clear();
				while( moves.size() < s ) 
					moves.add( curBest.getRandomMove() );
			}
			
			T localBest = null; // best local non-tabu individual
			TabuMove<T> localBestMove = null; 
			for (TabuMove<T> curMove : moves) { // get best localBest from curBest
				T cur = (T) curBest.applyMove(curMove);
				cur.setValue(eva.evaluate(cur));
				
				double penalty = 0;
				if( !rndMoveDiversication && freq.containsKey(curMove.getAttribute()) && noImpro > noImprosUntilPenalty && noImpro % noImprosUntilPenalty < maxPenaltyDuration )
					penalty = freq.get( curMove.getAttribute() ) * cur.getValue();

				if ((!tabuList.isTabu(curMove) && (localBest == null || cur.getValue() + penalty < localBest.getValue())) 
						|| cur.getValue() < globalBest.getValue() ) { // Aspiration by objecive
					localBest = cur;
					localBestMove = curMove;
				}
			}

			if (localBest == null) { // Aspiration by default, accept oldest move from tl
				for (TabuMove<T> curMove : moves) 
					// get and apply oldest move
					if( localBestMove == null || tabuList.getTenure(curMove) < tabuList.getTenure(localBestMove) ) {
						localBestMove = curMove;
					localBest = (T)curBest.applyMove(localBestMove);
					localBest.setValue(eva.evaluate(localBest));
				}
			}
			
			if (localBest.getValue() < globalBest.getValue()) {
				//log.debug("Found new global best: " + globalBest.getValue() + ", k: " + k);
				globalBest = localBest;

				freq.clear();
				noImpro = 0;
			} else
				noImpro++;

			curBest = localBest; // update curBest

			// update tabu list
			TabuMove<T> invMove = localBestMove.getInverse();
			if (!tabuList.isTabu(invMove))
				tabuList.add(invMove, tlLength); // tenure
			tabuList.step();
			
			// update freq table
			Object att = localBestMove.getAttribute();
			if( !freq.containsKey(att) )
				freq.put(att, 1);
			else
				freq.put(att,freq.get(att));	
		}
		return globalBest;
	}
}
