package regionalization.nga.tabu;

import java.util.Map.Entry;

import heuristics.tabu.TabuMove;

public class CutsTabuMove<T> implements TabuMove<T> {
		
	Edge<double[]> oldCut, newCut;
	
	public CutsTabuMove( Edge<double[]> oldCut, Edge<double[]> newCut ) {
		this.oldCut = oldCut;
		this.newCut = newCut;
	}
	
	public CutsTabuMove( Entry<double[],double[]> oldCut, Entry<double[],double[]> newCut ) {
		this.oldCut = new Edge<double[]>(oldCut);
		this.newCut = new Edge<double[]>(newCut);
	}

	public Entry<double[], double[]> getOldCut() {
		return oldCut.asEntry();
	}

	public Entry<double[], double[]> getNewCut() {
		return newCut.asEntry();
	}

	@Override
	public TabuMove<T> getInverse() {
		return new CutsTabuMove<>(newCut, oldCut);
	}

	@Override
	public Object getAttribute() {
		return oldCut;
	}
}
