package regionalization.tabu;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import regionalization.ga.ClusterCostCalculator;
import regionalization.ga.RegioGAIndividual;

public class RegioTabuIndividual extends RegioGAIndividual implements TabuIndividual{
	
	private Random r;

	public RegioTabuIndividual(List<double[]> genome, int seedSize, ClusterCostCalculator cc, Map<double[], Set<double[]>> cm) {
		super(genome, seedSize, cc, cm);
		this.r = new Random();
	}

	@Override
	public TabuIndividual applyMove(TabuMove tm) {
		RegioTabuMove m = (RegioTabuMove)tm;
		
		List<double[]> genome = new ArrayList<double[]>(getGenome());
		
		
		double[] valB = genome.set( m.getFrom(), genome.get( m.getTo() ) );
		genome.set( m.getTo(), valB );
		
		return new RegioTabuIndividual(genome, seedSize, cc, cm);
	}

	@Override
	public TabuMove getRandomMove() {
		int from = r.nextInt(getGenome().size());
		int to = r.nextInt(getGenome().size());
		
		if( from > to ) 
			return new RegioTabuMove( to, from );
		else
			return new RegioTabuMove( from, to);
	}

	@Override
	public List<TabuMove> getAllMoves() {
		List<TabuMove> moves = new ArrayList<TabuMove>();
		for( int i = 0; i < getGenome().size()-1; i++ )
			for( int j = i+1; j < getGenome().size(); j++ )
				moves.add( new RegioTabuMove( i, j) );
		return moves;
	}
}
