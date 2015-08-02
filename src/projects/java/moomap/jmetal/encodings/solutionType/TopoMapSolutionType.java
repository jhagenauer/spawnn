package moomap.jmetal.encodings.solutionType;

import moomap.jmetal.encodings.variable.TopoMap;
import jmetal.core.Problem;
import jmetal.core.SolutionType;
import jmetal.core.Variable;

public class TopoMapSolutionType extends SolutionType {
	
	int xDim, yDim, length;
	
	public TopoMapSolutionType( int xDim, int yDim, int length, Problem problem) throws ClassNotFoundException {
		super(problem) ;
		this.xDim = xDim;
		this.yDim = yDim;
		this.length = length;
	}
		
	@Override
	public Variable[] createVariables() {
	    return new Variable[]{ new TopoMap( xDim, yDim, length, problem_) } ;
	} 
		
	public Variable[] copyVariables(Variable[] vars) {
		return new Variable[]{ vars[0].deepCopy() };
	} 
}
