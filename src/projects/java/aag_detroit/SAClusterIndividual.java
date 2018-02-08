package aag_detroit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import heuristics.sa.SAIndividual;

public class SAClusterIndividual extends ClusterIndividual implements SAIndividual<SAClusterIndividual> {
		
	public SAClusterIndividual(List<Set<double[]>> l ) {
		super(l);
	}

	@Override
	public void step() {
		Collections.swap(l, r.nextInt(l.size()), r.nextInt(l.size()));
		return;
	}

	@Override
	public SAClusterIndividual getCopy() {
		return new SAClusterIndividual( new ArrayList<>(l));
	}
	
	public List<Set<double[]>> getList() {
		return l;
	}
}
