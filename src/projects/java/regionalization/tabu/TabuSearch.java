package regionalization.tabu;

import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

public class TabuSearch {
	
	private static Logger log = Logger.getLogger(TabuSearch.class);
		
	Random r = new Random();
	
	public int tlLength = 10;
	
	public TabuIndividual search( TabuIndividual init ) {

		TabuList<TabuMove> tl = new TabuList<TabuMove>(); // recency
				
		TabuIndividual curBest = init; // aktuell
		
		TabuIndividual globalBest = curBest; // global Best
		int kMax = 200; 
		int noImpro = 0;
						
		for( int k = 0; k < kMax; k++ ) {
			
			// get best non-tabu or global best move
			TabuIndividual localBest = null; // best local non-tabu
			TabuMove localBestMove = null;
						
			/*Set<TabuMove> moves = new HashSet<TabuMove>();
			while( moves.size() < 400 )
				moves.add( curBest.getRandomMove() );*/
			
			List<TabuMove> moves = curBest.getAllMoves();
			//log.debug("moves: "+moves.size() );
			
			for( TabuMove curMove : moves ) { // get best move for curBest
								
				TabuIndividual cur = curBest.applyMove( curMove );			
				if( ( !tl.isTabu( curMove ) && ( localBest == null || cur.getValue() < localBest.getValue() ) )
						|| cur.getValue() < globalBest.getValue() ) { // Aspiration by objecive
							localBest = cur;
							localBestMove = curMove;									
				}
			}
																							
			if( localBest == null ) //Aspiration by default, TODO: must be implemented
				log.error("Only tabu-moves possible!");

			log.debug(k+", localbest: "+localBest.getValue() );
											
			if( localBest.getValue() < globalBest.getValue() ) { 
				log.debug("Found new global best: "+globalBest.getValue()+", k: "+k );	
				globalBest = localBest;
				noImpro = 0;
			} else
				noImpro++;
								
			curBest = localBest; // update curBest
			TabuMove invMove = localBestMove; 
			if( !tl.isTabu( invMove ) ) 
				tl.add( invMove, tlLength ); // tenure
						
			tl.step();
		}
		return globalBest;
	}
}
