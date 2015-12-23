package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.FeatureDescriptor;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.ListCellRenderer;

import net.miginfocom.swing.MigLayout;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.FeatureStore;
import org.geotools.data.FileDataStoreFactorySpi;
import org.geotools.data.shapefile.ShapefileDataStoreFactory;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.feature.type.Name;

import spawnn.dist.Dist;
import spawnn.gui.ClusterDialogGrid.ClusterAlgorithm;
import spawnn.gui.DistanceDialog.DistMode;
import spawnn.gui.DistanceDialog.StatMode;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomToolboxUtils;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.ColorUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.Clustering.TreeNode;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;

public class SOMResultPanel extends ResultPanel<GridPos> implements ActionListener, NeuronSelectedListener<GridPos> {

	private static Logger log = Logger.getLogger(SOMResultPanel.class);
	private static final long serialVersionUID = -4518072006960672609L;

	private JComboBox<String> gridComboBox;
	private JComboBox colorComboBox, gridModeComboBox;
	private JButton btnExpGrid, btnExpMap, colorChooser;
	private GridPanel pnlGrid;
	private GraphPanel pnlGraph;
	private JPanel cards;
	MapPanel<GridPos> mapPanel;

	private JToggleButton selectSingle;

	private Grid2D<double[]> grid;
	private Graph<double[], double[]> graph;

	private List<GridPos> pos;
	private List<double[]> samples;
	private Map<GridPos, Set<double[]>> bmus;
	private Dist<double[]> fDist, gDist;

	private Map<GridPos, Double> neuronValues;
	private Map<GridPos, Color> selectedColors = new HashMap<GridPos, Color>();

	private FeatureCollection<SimpleFeatureType, SimpleFeature> fc;
	private List<String> names;

	private static final String RANDOM = "Random", DISTANCE = "Distance...", CLUSTER = "Cluster...";
	private static final String GRID = "SOM", GRAPH = "Neurons (Geo)";

	private Frame parent;

	public SOMResultPanel(Frame parent, SpatialDataFrame orig, List<double[]> samples, Map<GridPos, Set<double[]>> bmus, Grid2D<double[]> grid, Dist<double[]> fDist, Dist<double[]> gDist, int[] fa, int[] ga, boolean wmc) {
		super();

		this.pos = new ArrayList<GridPos>(grid.getPositions()); // fixed order for indexing/coloring
		this.fc = buildClusterFeatures(orig, samples, bmus, pos);
		this.grid = grid;
		this.bmus = bmus;
		this.fDist = fDist;
		this.gDist = gDist;
		this.names = orig.names;
		this.parent = parent;

		setLayout(new MigLayout(""));
		
		gridComboBox = new JComboBox<String>();
		gridComboBox.addItem(RANDOM);
		gridComboBox.addItem(DISTANCE);
		gridComboBox.addItem(CLUSTER);
		
		gridComboBox.setRenderer(new ComboSeparatorsRenderer((ListCellRenderer<String>)gridComboBox.getRenderer()){        
		    @Override
			protected boolean addSeparatorAfter(JList list, String value, int index) {
		    	return CLUSTER.equals(value);
			}                                                                            
		});     
		
		Set<Integer> fas = new HashSet<Integer>();
		for( int i : fa )
			fas.add(i);
		for( int i = 0; i < orig.names.size(); i++ ) {
			String s = orig.names.get(i);
			if( fas.contains(i))
				s+="*";
			gridComboBox.addItem(s);
		}
		if( wmc )
			for( int i = 0; i < orig.names.size(); i++ ) {
				String s = orig.names.get(i);
				if( fas.contains(i))
					s+="*";
				s+= " (ctx)";
				gridComboBox.addItem(s);
			}
		gridComboBox.addActionListener(this);

		colorComboBox = new JComboBox();
		colorComboBox.setModel(new DefaultComboBoxModel(ColorUtils.ColorMode.values()));
		colorComboBox.addActionListener(this);

		gridModeComboBox = new JComboBox();
		gridModeComboBox.addItem(GRID);
		if (ga != null && ga.length == 2) {
			gridModeComboBox.addItem(GRAPH);
			gridModeComboBox.addActionListener(this);
			gridModeComboBox.setEnabled(true);
		} else {
			gridModeComboBox.setEnabled(false);
		}

		btnExpGrid = new JButton("Export grid...");
		btnExpGrid.addActionListener(this);

		colorChooser = new JButton("Select color...");
		colorChooser.setBackground(selectedColor);
		colorChooser.addActionListener(this);

		selectSingle = new JToggleButton("Select single");
		selectSingle.addActionListener(this);

		btnExpMap = new JButton("Export map...");
		btnExpMap.addActionListener(this);

		graph = new UndirectedSparseGraph<double[], double[]>();
		for (GridPos gp : grid.getPositions()) {
			double[] a = grid.getPrototypeAt(gp);
			if (!graph.getVertices().contains(a))
				graph.addVertex(a);

			for (GridPos nb : grid.getNeighbours(gp)) {
				double[] b = grid.getPrototypeAt(nb);
				if (!graph.getVertices().contains(b))
					graph.addVertex(b);

				if (gDist != null)
					graph.addEdge(new double[] { fDist.dist(a, b), gDist.dist(a, b) }, a, b);
				else
					graph.addEdge(new double[] { fDist.dist(a, b) }, a, b);
			}
		}

		pnlGrid = new GridPanel(grid, fDist);
		mapPanel = new MapPanel<GridPos>(fc, pos);
		pnlGraph = new GraphPanel(graph, ga);

		if (ga != null && ga.length == 2) {
			pnlGraph.setGraphLayout(GraphPanel.Layout.Geo);
		}

		// maybe it would be nicer to use a MapPanel and to draw the network on a "real" map
		class NSL implements NeuronSelectedListener<double[]> {
			Grid2D<double[]> grid;

			NSL(Grid2D<double[]> grid) {
				this.grid = grid;
			}

			@Override
			public void neuronSelectedOccured(NeuronSelectedEvent<double[]> evt) {
				GridPos gp = grid.getPositionOf(evt.getNeuron());

				if (selectedColors.containsKey(gp) && selectedColors.get(gp) == selectedColor)
					selectedColors.remove(gp);
				else
					selectedColors.put(gp, selectedColor);

				Map<GridPos, Color> colorMap = ColorUtils.getColorMap(neuronValues, (ColorUtils.ColorMode) colorComboBox.getSelectedItem());
				pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
				mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
				pnlGraph.setGridColors(toDoubleArrayMap(colorMap), toDoubleArrayMap(selectedColors), toDoubleArrayMap(neuronValues));
			}
		}
		pnlGraph.addNeuronSelectedListener(new NSL(grid));

		actionPerformed(new ActionEvent(gridComboBox, 0, DISTANCE));
		Map<GridPos, Color> colorMap = ColorUtils.getColorMap(neuronValues, (ColorUtils.ColorMode) colorComboBox.getSelectedItem());

		pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
		pnlGrid.addMouseListener(pnlGrid);
		pnlGrid.addNeuronSelectedListener(this);

		mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
		mapPanel.addNeuronSelectedListener(this);

		cards = new JPanel(new CardLayout());
		cards.add(pnlGrid, GRID);
		cards.add(pnlGraph, GRAPH);

		add(gridComboBox, "split 4");
		add(colorComboBox, "");
		add(gridModeComboBox, "");
		add(btnExpGrid, "");
				
		add(colorChooser, "split 3");
		add(selectSingle, "");
		add(btnExpMap, "pushx, wrap");
		
		add( cards, "w 50%, pushy, grow");
		add( mapPanel, "grow");
		
		colorComboBox.setSelectedItem(ColorUtils.ColorMode.Blues);
	}

	private Color selectedColor = Color.RED;

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == gridComboBox) { // som-visualization-change
			if (gridComboBox.getSelectedItem() == DISTANCE) { // dmatrix

				DistanceDialog dd = new DistanceDialog(parent, "Distance...", true, gDist != null);
				DistMode dm = dd.getDistMode();
				StatMode sm = dd.getStatMode();

				neuronValues = new HashMap<GridPos, Double>();
				for (GridPos p : grid.getPositions()) {
					double[] v = grid.getPrototypeAt(p);

					DescriptiveStatistics ds = new DescriptiveStatistics();
					for (GridPos np : grid.getNeighbours(p)) {
						if (dm == DistMode.Normal)
							ds.addValue(fDist.dist(v, grid.getPrototypeAt(np)));
						else
							ds.addValue(gDist.dist(v, grid.getPrototypeAt(np)));
					}
					if (sm == StatMode.Mean)
						neuronValues.put(p, ds.getMean());
					else if (sm == StatMode.Median)
						neuronValues.put(p, ds.getPercentile(0.5));
					else if (sm == StatMode.Variance)
						neuronValues.put(p, ds.getVariance());
					else if (sm == StatMode.Min)
						neuronValues.put(p, ds.getMin());
					else if (sm == StatMode.Max)
						neuronValues.put(p, ds.getMax());
				}
			} else if (gridComboBox.getSelectedItem() == RANDOM) { // random
				int k = 0;
				List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
				Collections.shuffle(pos);
				neuronValues = new HashMap<GridPos, Double>();
				for (GridPos p : pos)
					neuronValues.put(p, (double) k++);
			} else if (gridComboBox.getSelectedItem() == CLUSTER) { // cluster
				ClusterDialogGrid cd = new ClusterDialogGrid(parent, CLUSTER, true, true);

				if (cd.isOkPressed()) {
					parent.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

					List<double[]> prototypes = new ArrayList<double[]>(grid.getPrototypes());
					List<Set<double[]>> clusters = null;

					// connected map
					Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
					if (cd.getConnected()) {
						for (GridPos p : grid.getPositions()) {
							double[] v = grid.getPrototypeAt(p);
							Set<double[]> s = new HashSet<double[]>();
							for (GridPos nb : grid.getNeighbours(p))
								s.add(grid.getPrototypeAt(nb));
							cm.put(v, s);
						}
					}

					if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.kMeans)
						clusters = new ArrayList<Set<double[]>>(Clustering.kMeans(prototypes, cd.getNumCluster(), fDist).values());
					else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SKATER) {
						Map<double[], Set<double[]>> mst = Clustering.getMinimumSpanningTree(cm, fDist);
						clusters = Clustering.skater(mst, cd.getNumCluster() - 1, fDist, 1);
					} else if (cd.getAlgorithm() == ClusterAlgorithm.Watershed) {
						Collection<Set<GridPos>> wsc;
						if (grid instanceof Grid2DHex) {
							wsc = SomUtils.getWatershedHex(cd.getMinimum(), cd.getMaximum(), cd.getBlur(), grid, fDist, false);
						} else { // TODO not sure if the implementation for non-hex-maps is correct
							int[][] ws = SomUtils.getWatershed(cd.getMinimum(), cd.getMaximum(), cd.getBlur(), grid, fDist, false);
							wsc = SomUtils.getClusterFromWatershed(ws, grid);
						}
						clusters = new ArrayList<Set<double[]>>();

						for (Set<GridPos> c : wsc) {
							Set<double[]> l = new HashSet<double[]>();
							for (GridPos p : c)
								l.add(grid.getPrototypeAt(p));
							clusters.add(l);
						}
					} else { // hierarchical
						Clustering.HierarchicalClusteringType type = null;
						if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.ALK)
							type = Clustering.HierarchicalClusteringType.average_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.CLK)
							type = Clustering.HierarchicalClusteringType.complete_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SLK)
							type = Clustering.HierarchicalClusteringType.single_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.Ward)
							type = Clustering.HierarchicalClusteringType.ward;
						
						log.debug("Algorithm: "+type+","+cd.getConnected() );
						Map<Set<double[]>,TreeNode> tree;
						if (cd.getConnected()) 
							tree = Clustering.getHierarchicalClusterTree(cm, fDist, type);
						else
							tree = Clustering.getHierarchicalClusterTree(prototypes, fDist, type);
						clusters = Clustering.cutTree( tree, cd.getNumCluster() );
					}

					// prototypes to samples
					Map<double[], Set<double[]>> nBmus = new HashMap<double[], Set<double[]>>();
					for (double d[] : prototypes) {
						GridPos p = grid.getPositionOf(d);
						if (bmus.containsKey(p))
							nBmus.put(d, bmus.get(p));
						else
							nBmus.put(d, new HashSet<double[]>());
					}

					Map<double[], Set<double[]>> ll = ResultPanel.prototypeClusterToDataCluster(nBmus, clusters);
					log.debug("#Cluster: " + ll.size());
					log.debug("Within clusters sum of squares: " + ClusterValidation.getWithinClusterSumOfSuqares(ll.values(), fDist));
					log.debug("Between clusters sum of squares: " + ClusterValidation.getBetweenClusterSumOfSuqares(ll.values(), fDist));
					// log.debug("Connectivity: " + ClusterValidation.getConnectivity(ll, fDist, 10));
					log.debug("Dunn Index: " + ClusterValidation.getDunnIndex(ll.values(), fDist));

					log.debug("quantization error: " + DataUtils.getMeanQuantizationError(ll, fDist));
					if( gDist != null )
					log.debug("spatial quantization error: " + DataUtils.getMeanQuantizationError(ll, gDist));
					log.debug("Davies-Bouldin Index: " + ClusterValidation.getDaviesBouldinIndex(ll, fDist));
					log.debug("Silhouette Coefficient: " + ClusterValidation.getSilhouetteCoefficient(ll, fDist));

					// means, because cluster summarize multiple prototypes
					List<double[]> means = new ArrayList<double[]>(ll.keySet());
					if( gDist != null )
					Collections.sort(means, new Comparator<double[]>() {
						@Override
						public int compare(double[] o1, double[] o2) {
							double[] d = new double[o1.length];
							if (gDist.dist(o1, d) < gDist.dist(o2, d))
								return -1;
							else if (gDist.dist(o1, d) > gDist.dist(o2, d))
								return 1;
							else
								return 0;
						}
					});
																		
					neuronValues = new HashMap<GridPos, Double>();
			
					for( Set<double[]> s : clusters ) {
						// get data mapped by all prototypes in s
						Set<double[]> data = new HashSet<double[]>();
						for( double[] proto : s )
							data.addAll( nBmus.get(proto) );
							
						// search mean by data
						double[] mean = null;
						for( Entry<double[],Set<double[]>> en : ll.entrySet()  )
							if( en.getValue().containsAll(data) && data.containsAll(en.getValue())) {
								mean = en.getKey();
								break;
							}
													
						// color protos of s
						double v = means.indexOf(mean);
						for( double[] proto : s ) 
							neuronValues.put(grid.getPositionOf(proto), v);				
					}
					
					parent.setCursor(Cursor.getDefaultCursor());
				}
			} else { // components
				for (GridPos p : grid.getPositions()) {
					double[] v = grid.getPrototypeAt(p);
					neuronValues.put(p, v[gridComboBox.getSelectedIndex() - 3]); // RANDOM, DISTANCE, CLUSTER
				}
			}

			Map<GridPos, Color> colorMap = ColorUtils.getColorMap(neuronValues, (ColorUtils.ColorMode) colorComboBox.getSelectedItem());
			pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
			mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
			pnlGraph.setGridColors(toDoubleArrayMap(colorMap), toDoubleArrayMap(selectedColors), toDoubleArrayMap(neuronValues));
			// selected.clear(); // reset selected if other vis8

		} else if (e.getSource() == colorComboBox) { // color-mode-change
			Map<GridPos, Color> colorMap = ColorUtils.getColorMap(neuronValues, (ColorUtils.ColorMode) colorComboBox.getSelectedItem());
			pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
			mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
			pnlGraph.setGridColors(toDoubleArrayMap(colorMap), toDoubleArrayMap(selectedColors), toDoubleArrayMap(neuronValues));
		} else if (e.getSource() == gridModeComboBox) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			cl.show(cards, (String) gridModeComboBox.getSelectedItem());
		} else if (e.getSource() == btnExpGrid) {
			JFileChooser fc = new JFileChooser("output");

			fc.setFileFilter(FFilter.unitFilter);
			fc.setFileFilter(FFilter.weightFilter);
			fc.setFileFilter(FFilter.pngFilter);
			fc.setFileFilter(FFilter.epsFilter);
			fc.setFileFilter(FFilter.somXMLFilter);
			fc.setFileFilter(FFilter.graphMLFilter);

			int state = fc.showSaveDialog(this);
			if (state == JFileChooser.APPROVE_OPTION) {
				File fn = fc.getSelectedFile();
				try {
					if (fc.getFileFilter() == FFilter.pngFilter) {
						if (gridModeComboBox.getSelectedItem() == GRID)
							pnlGrid.saveImage(fn, "PNG");
						else
							pnlGraph.saveImage(fn, "PNG");
					} else if (fc.getFileFilter() == FFilter.epsFilter) {
						String s = fn.getAbsolutePath(); //TODO remove this
						s = s.replaceFirst(".eps", "_legend.eps");
						//saveLegend(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File(s), "EPS");
						
						if (gridModeComboBox.getSelectedItem() == GRID)
							pnlGrid.saveImage(fn, "EPS");
						else
							pnlGraph.saveImage(fn, "EPS");
					} else if (fc.getFileFilter() == FFilter.somXMLFilter) {
						SomUtils.saveGrid(grid, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.unitFilter) {
						SomToolboxUtils.writeUnitDescriptions(grid, samples, bmus, fDist, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.weightFilter) {
						SomToolboxUtils.writeWeightVectors(grid, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.graphMLFilter) {
						Map<double[], Double> nNV = new HashMap<double[], Double>();
						for (GridPos gp : neuronValues.keySet())
							nNV.put(grid.getPrototypeAt(gp), neuronValues.get(gp));
						NGResultPanel.writeGraphToGraphML(names, graph, nNV, toDoubleArrayMap(selectedColors), fn);
					}
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}
			}
		} else if (e.getSource() == colorChooser) {
			selectedColor = JColorChooser.showDialog(this, "Select selection color", selectedColor);
			colorChooser.setBackground(selectedColor);		
		} else if (e.getSource() == btnExpMap) {
			JFileChooser fChoser = new JFileChooser("output");

			fChoser.setFileFilter(FFilter.shpFilter);
			fChoser.setFileFilter(FFilter.epsFilter);
			fChoser.setFileFilter(FFilter.pngFilter);

			int state = fChoser.showSaveDialog(this);
			if (state == JFileChooser.APPROVE_OPTION) {
				File fn = fChoser.getSelectedFile();
				if (fChoser.getFileFilter() == FFilter.pngFilter) {
					mapPanel.saveImage(fn, "PNG");
				} else if (fChoser.getFileFilter() == FFilter.epsFilter) {
					mapPanel.saveImage(fn, "EPS");
				} else if (fChoser.getFileFilter() == FFilter.shpFilter) {
					try {
						// ugly but works
						FeatureIterator<SimpleFeature> fit = fc.features();
						while (fit.hasNext()) {
							SimpleFeature sf = (SimpleFeature) fit.next();
							GridPos gp = pos.get((Integer) (sf.getAttribute("neuron")));
							sf.setAttribute("nValue", neuronValues.get(gp));
							if (selectedColors.containsKey(gp))
								sf.setAttribute("selColor", selectedColors.get(gp) );
						}
						fit.close();
						
						Map map = Collections.singletonMap("url", fn.toURI().toURL());
						FileDataStoreFactorySpi factory = new ShapefileDataStoreFactory();
						DataStore myData = factory.createNewDataStore(map);
						myData.createSchema(fc.getSchema());
						Name name = myData.getNames().get(0);
						FeatureStore<SimpleFeatureType, SimpleFeature> store = (FeatureStore<SimpleFeatureType, SimpleFeature>) myData.getFeatureSource(name);
						store.addFeatures(fc);
					} catch (Exception ex) {
						ex.printStackTrace();
					}
				}

			}
		} else if (e.getSource() == selectSingle) {
			if (!mapPanel.selectSingle)
				mapPanel.selectSingle = true;
			else
				mapPanel.selectSingle = false;
		}
	}

	@Override
	public void neuronSelectedOccured(NeuronSelectedEvent<GridPos> evt) {
		GridPos gp = evt.getNeuron();

		if (selectedColors.containsKey(gp) && selectedColors.get(gp) == selectedColor)
			selectedColors.remove(gp);
		else
			selectedColors.put(gp, selectedColor);

		Map<GridPos, Color> colorMap = ColorUtils.getColorMap(neuronValues, (ColorUtils.ColorMode) colorComboBox.getSelectedItem());
		pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
		mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
		pnlGraph.setGridColors(toDoubleArrayMap(colorMap), toDoubleArrayMap(selectedColors), toDoubleArrayMap(neuronValues));
	}

	private <T> Map<double[], T> toDoubleArrayMap(Map<GridPos, T> cm) {
		Map<double[], T> nm = new HashMap<double[], T>();
		for (GridPos gp : cm.keySet())
			nm.put(grid.getPrototypeAt(gp), cm.get(gp));
		return nm;
	}
}
