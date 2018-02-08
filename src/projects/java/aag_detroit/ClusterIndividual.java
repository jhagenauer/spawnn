package aag_detroit;

import java.util.List;
import java.util.Random;
import java.util.Set;

import heuristics.HeuristicsIndividual;

public class ClusterIndividual implements HeuristicsIndividual {
	Random r = new Random();
	List<Set<double[]>> l;
	
	public ClusterIndividual(List<Set<double[]>> l ) {
		this.l = l;
	}
	
	public List<Set<double[]>> getList() {
		return l;
	}
}
