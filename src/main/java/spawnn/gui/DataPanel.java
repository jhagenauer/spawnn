package spawnn.gui;

import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.DefaultCellEditor;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import net.miginfocom.swing.MigLayout;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class DataPanel extends JPanel implements ActionListener, TableModelListener {
	
	private static Logger log = LogManager.getLogger(DataPanel.class);
	private static final long serialVersionUID = 1736311423918203189L;
	public static final String TRAIN_ALLOWED_PROP ="TRAIN_ALLOWED"; //TODO maybe better use training-events instead of property changes
	static Set<String> coordNames;
	private boolean useAllState = false;
			
	static {
		coordNames = new HashSet<String>();
		coordNames.add("x");
		coordNames.add("X");
		coordNames.add("y");
		coordNames.add("Y");
		coordNames.add("lon");
		coordNames.add("LON");
		coordNames.add("Lon");
		coordNames.add("longitude");
		coordNames.add("lat");
		coordNames.add("LAT");
		coordNames.add("Lat");
		coordNames.add("latitude");
	}
		
	private DefaultTableModel dataTable;
	private JButton btnLoadData, addCCoords, useAll, btnDistMat;
	private BoxPlotPanel boxPlotPnl;
	private JTextField infoField;
	
	enum norm {none, scale01, zScore};
	final int ATTRIBUTE = 0, NORM = 1, COORDINATE = 2, USE = 3;
	
	protected SpatialDataFrame sdf = null;
	protected Map<double[], Map<double[], Double>> dMap = null;
	
	private Frame parent;
	
	DataPanel(Frame parent) {
		super();
		this.parent = parent;
		
		setLayout(new MigLayout(""));
		
		btnLoadData = new JButton("Load data...");
		btnLoadData.addActionListener(this);
		
		addCCoords = new JButton("Add centroid coordinates");
		addCCoords.addActionListener(this);
		addCCoords.setEnabled(false);
				
		useAll = new JButton("Use all");
		useAll.addActionListener(this);
		useAll.setEnabled(false);
		
		btnDistMat = new JButton("Dist. matrix...");
		btnDistMat.addActionListener(this);
		btnDistMat.setEnabled(false);
						
		dataTable = new DefaultTableModel( new String[]{"Attribute","Normalize","Coordinate","Use"}, 0 ) {
			private static final long serialVersionUID = -78686917074445663L;

			@Override
			public Class<?> getColumnClass(int column) {
				switch (column) {
                case ATTRIBUTE:
                    return String.class;
                case NORM:
                    return String.class;
                case COORDINATE:
                    return Boolean.class;
                case USE:
                    return Boolean.class;
                default:
                	return String.class;
				}
			}
		};
		
		dataTable.addTableModelListener(this);
		
		final JTable table = new JTable(dataTable);
		table.getColumnModel().getColumn(1).setCellEditor(new DefaultCellEditor( new JComboBox(norm.values())));
		
		table.getColumnModel().getColumn(0).setPreferredWidth(250);
		table.getColumnModel().getColumn(1).setPreferredWidth(20);
		table.getColumnModel().getColumn(2).setPreferredWidth(10);
		table.getColumnModel().getColumn(3).setPreferredWidth(10);
				
		boxPlotPnl = new BoxPlotPanel();
		
		add(btnLoadData, "split 5");
		add(addCCoords, "");
		add(btnDistMat,"");
		add(useAll,"wrap");
		
		add( new JScrollPane(table), "w 50%, grow" );
		add( boxPlotPnl, "w 50%, push, grow,wrap");
		
		infoField = new JTextField();
		infoField.setEditable(false);
		updateStatus();
		add( infoField,"span 2, growx");
	}
	
	private File lastDir = null;

	@Override
	public void actionPerformed(ActionEvent ae) {
		
		if( ae.getSource() == btnLoadData ) {
			
			JFileChooser fc = new JFileChooser();
			fc.setFileFilter(FFilter.csvFilter); 
			fc.setFileFilter(FFilter.shpFilter);
			
			if( lastDir != null )
				fc.setCurrentDirectory(lastDir);
									
			int state = fc.showOpenDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) {
				parent.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
				
				dMap = null;
				normedSamples = null;
				File fn = fc.getSelectedFile();
				lastDir = fc.getCurrentDirectory();
				useAllState = false;
				
				if( fc.getFileFilter() == FFilter.shpFilter ) {
			      sdf = DataUtils.readSpatialDataFrameFromShapefile(fn, true); 
			      addCCoords.setEnabled(true);
				} else if( fc.getFileFilter() == FFilter.csvFilter ) {		
					DataFrame df = DataUtils.readDataFrameFromCSV(fn, new int[]{}, true ); 
					sdf = new SpatialDataFrame();				
					sdf.samples = df.samples;
					sdf.names = df.names;
					sdf.bindings = df.bindings;
					
					addCCoords.setEnabled(false);
				}
				
				if( sdf.samples.isEmpty() ) {
					JOptionPane.showMessageDialog(this, "No rows in data file.");
					return;
				}
									
				useAll.setEnabled(true);
				dataTable.setRowCount(0);
								
				for( String s : sdf.names )
			    	if( coordNames.contains(s) )
			    		dataTable.addRow( new Object[]{s, norm.none, true, true } );
			    	else
			    		dataTable.addRow( new Object[]{s, norm.zScore, false, false } );
			    			    
				boxPlotPnl.setData(sdf.names, getNormedSamples() );
				boxPlotPnl.plot();	
				btnDistMat.setEnabled(true);
				updateStatus();

				parent.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
			}
		} else if( ae.getSource() == addCCoords ) {
			dMap = null; 
			normedSamples = null;
			
			List<double[]> ns = new ArrayList<double[]>();
			for( int i = 0; i < sdf.samples.size(); i++ ) {
			    double[] a = sdf.samples.get(i); 
				double[] b = Arrays.copyOf(a, a.length+2);
				b[a.length] = sdf.geoms.get(i).getCentroid().getX();
				b[a.length+1] = sdf.geoms.get(i).getCentroid().getY();
				ns.add(b);
			}
			sdf.samples = ns;
			
			sdf.names.add("x");
			sdf.names.add("y");
			
			sdf.bindings.add(SpatialDataFrame.binding.Double);
			sdf.bindings.add(SpatialDataFrame.binding.Double);
			
			// get current norm-mode
			norm n = norm.none;
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) {
					n = (norm) dataTable.getValueAt(i, NORM);
					break;
				}
			}
							
			dataTable.addRow( new Object[]{"x", n, true, true } );
			dataTable.addRow( new Object[]{"y", n, true, true } );
						
			addCCoords.setEnabled(false);
			
			boxPlotPnl.setData(sdf.names, getNormedSamples() );
			boxPlotPnl.plot();
		} else if ( ae.getSource() == useAll ) {
			for( int i = 0; i < dataTable.getRowCount(); i++ )
				dataTable.setValueAt( !useAllState, i, USE );
			useAllState = !useAllState;
		} else if( ae.getSource() == btnDistMat ) {
			DistMatrixDialog dmd = new DistMatrixDialog(parent, "Create dist. matrix", true, getNormedSamples(), sdf, getGA(true));
			if( dmd.okPressed() )
				dMap = dmd.getDistanceMap();
			else
				dMap = null;
			updateStatus();
		}
	}
	
	private boolean trainAllowed = false;
			
	@Override
	public void tableChanged(TableModelEvent e) {		
		if( e.getColumn() == COORDINATE || e.getColumn() == USE ) {
			boolean a = getFA().length > 0 && ( sdf.geoms != null || getGA(true).length > 0 );
			firePropertyChange(TRAIN_ALLOWED_PROP, trainAllowed, a );
			trainAllowed = a;
		}
		
		dataTable.removeTableModelListener(this);
		// all used geoColumns must have the same normalization
		if( e.getColumn() == COORDINATE && (Boolean)dataTable.getValueAt(e.getFirstRow(), COORDINATE) ) {	
			// Set norm to the one of the others
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( i == e.getFirstRow() )
					continue;
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) {
					dataTable.setValueAt( dataTable.getValueAt(i, NORM), e.getFirstRow(), NORM );
					break;
				}
			}
		} else if( e.getColumn() == NORM && (Boolean)dataTable.getValueAt(e.getFirstRow(), COORDINATE) ) {
			// change norm of all other
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( i == e.getFirstRow() ) //FIXME i is integer!!!
					continue;
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) 
					dataTable.setValueAt( dataTable.getValueAt(e.getFirstRow(), NORM), i, NORM );
			}
		}
		dataTable.addTableModelListener(this);
		
		boxPlotPnl.setData(sdf.names, getNormedSamples() );
		boxPlotPnl.plot();
	}
	
	private List<double[]> normedSamples = null;
	public List<double[]> getNormedSamples() {
		if( normedSamples == null ) {
			normedSamples = new ArrayList<double[]>();
			for( double[] d : sdf.samples )
				normedSamples.add( Arrays.copyOf(d, d.length));
		} else {
			for( int i = 0; i < normedSamples.size(); i++ )
				for( int j = 0; j < normedSamples.get(0).length; j++ )
					normedSamples.get(i)[j] = sdf.samples.get(i)[j];
		}
		
		boolean geoNormed = false;
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( !(Boolean)dataTable.getValueAt(i, COORDINATE) ) {
				if( dataTable.getValueAt(i, NORM) == norm.scale01 )
					DataUtils.normalizeColumn(normedSamples, i);
				else if( dataTable.getValueAt(i, NORM) == norm.zScore )
					DataUtils.zScoreColumn(normedSamples, i);
			} else if( !geoNormed && (Boolean)dataTable.getValueAt(i, COORDINATE) ) { // only once
				int[] ga = getGA(true);
				if( dataTable.getValueAt(i, NORM) == norm.scale01 )
					DataUtils.normalizeGeoColumns(normedSamples, ga );
				else if( dataTable.getValueAt(i, NORM) == norm.zScore  )
					DataUtils.zScoreGeoColumns(normedSamples, ga, new EuclideanDist(ga) );
				geoNormed = true;	
			}
		}		
		return normedSamples;
	}
	
	public SpatialDataFrame getSpatialData() {
		if( sdf.geoms == null ) { // no geoms present, if data stems from csv
			GeometryFactory gf = new GeometryFactory();
			sdf.geoms = new ArrayList<Geometry>();
			
			int[] ga = getGA(true);
						
			for( double[] d : sdf.samples ) {
				Coordinate c;
				if( ga.length == 2 )
					c = new Coordinate(d[ga[0]],d[ga[1]]);
				else if( ga.length == 3 )
					c = new Coordinate(d[ga[0]],d[ga[1]]);
				else 
					throw new RuntimeException("No spatial data submitted/found!");
				
				sdf.geoms.add( gf.createPoint(c) );
			}
		}
		return sdf;
	}
		
	public int[] getFA() {
		List<Integer> l = new ArrayList<Integer>();
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( (Boolean)dataTable.getValueAt(i, USE) && !(Boolean)dataTable.getValueAt(i, COORDINATE) )
				l.add(i);
		}
		int[] fa = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			fa[i] = l.get(i);
		return fa;
	}
	
	// Returns either the used GA or the complete GA, depending on flag
	public int[] getGA(boolean allGA) {
		List<Integer> l = new ArrayList<Integer>();
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( ( allGA || (Boolean)dataTable.getValueAt(i, USE) ) && (Boolean)dataTable.getValueAt(i, COORDINATE) )
				l.add(i);
		}		
		int[] ga = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			ga[i] = l.get(i);
		return ga;
	}
	
	public Map<double[],Map<double[],Double>> getDistanceMap() {
		return dMap;
	}
	
	public void updateStatus() {
		String a = "No data";
		String b = "No dist matrix";
		if( sdf != null && sdf.samples != null )
			a = sdf.samples.size()+" observations, "+sdf.names.size()+" attributes";
		if( dMap != null ) {
			int numEntries = 0;
			for( Map<double[],Double> m : dMap.values() )
				numEntries += m.size();
			b = numEntries + " entries";
		}
		infoField.setText(a+" / "+b);
	}
}
