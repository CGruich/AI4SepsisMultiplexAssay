function varargout = D3_Reconstruction_UI_Ver1c_Breast(varargin)
% D3_RECONSTRUCTION_UI_VER1C_BREAST MATLAB code for D3_Reconstruction_UI_Ver1c_Breast.fig
%      D3_RECONSTRUCTION_UI_VER1C_BREAST, by itself, creates a new D3_RECONSTRUCTION_UI_VER1C_BREAST or raises the existing
%      singleton*.
%
%      H = D3_RECONSTRUCTION_UI_VER1C_BREAST returns the handle to a new D3_RECONSTRUCTION_UI_VER1C_BREAST or the handle to
%      the existing singleton*.
%
%      D3_RECONSTRUCTION_UI_VER1C_BREAST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in D3_RECONSTRUCTION_UI_VER1C_BREAST.M with the given input arguments.
%
%      D3_RECONSTRUCTION_UI_VER1C_BREAST('Property','Value',...) creates a new D3_RECONSTRUCTION_UI_VER1C_BREAST or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before D3_Reconstruction_UI_Ver1c_Breast_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to D3_Reconstruction_UI_Ver1c_Breast_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help D3_Reconstruction_UI_Ver1c_Breast

% Last Modified by GUIDE v2.5 05-Mar-2017 15:25:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @D3_Reconstruction_UI_Ver1c_Breast_OpeningFcn, ...
                   'gui_OutputFcn',  @D3_Reconstruction_UI_Ver1c_Breast_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before D3_Reconstruction_UI_Ver1c_Breast is made visible.
function D3_Reconstruction_UI_Ver1c_Breast_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to D3_Reconstruction_UI_Ver1c_Breast (see VARARGIN)

% Choose default command line output for D3_Reconstruction_UI_Ver1c_Breast

%Create tab group
handles.tgroup = uitabgroup('Parent', handles.figure1,'TabLocation', 'left');
handles.tab1 = uitab('Parent', handles.tgroup, 'Title', 'Breast Cancer');
handles.tab2 = uitab('Parent', handles.tgroup, 'Title', 'IL3');
%Place panels into each tab
set(handles.uipanel1,'Parent',handles.tab1)
set(handles.uipanel2,'Parent',handles.tab2)
%Reposition each panel to same location as panel 1
set(handles.uipanel2,'position',get(handles.uipanel1,'position'));



handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes D3_Reconstruction_UI_Ver1c_Breast wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = D3_Reconstruction_UI_Ver1c_Breast_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;





% --- Executes on button press in LoadImages.
function LoadImages_Callback(hObject, eventdata, handles)
% hObject    handle to LoadImages (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.listbox1,'String',[]);

% First get the folder name
directory_name = uigetdir('','Choose a folder to import...');


FileExtension = get(handles.FileType,'String');   

theFiles = dir([directory_name,'/*.',FileExtension]);

numFiles = size(theFiles,1);
names = struct2cell(theFiles);
set(handles.listbox1,'String',names(1,:));
set(handles.text_dir,'String',[directory_name,'/']);

% Show the first image in the list
axes(handles.axes1)
imagesc(imread([directory_name,'/',theFiles(1).name]),[0,2^16]);
colormap('gray')
axis equal

ImageAxis = cell(numFiles-1,3);
handles.ImageAxis = ImageAxis;

guidata(hObject, handles);

               


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1

contents = get(hObject,'String');

xlimit = get(handles.axes1,'Xlim');
ylimit = get(handles.axes1,'Ylim');

minx = ceil(xlimit(1));
maxx = floor(xlimit(2));
miny = ceil(ylimit(1));
maxy = floor(ylimit(2));

if minx<1
    minx = 1;
end

if miny<1
    miny = 1;
end

%2592x1944 old // 2560x1920 new
if maxx>2592
    maxx = 2592;
end

if maxy>1944
    maxy = 1944;
end

data = double(imread([get(handles.text_dir,'String'),contents{get(hObject,'Value')}]));

axes(handles.axes1)

% Show the normalized data if a reference image is selected. 
RefLoaded = get(handles.text_ref,'String');
if strcmp(RefLoaded,'Not loaded')
    imagesc(data)
else
    % When backround intensities are different between sample and ref images
    ref = handles.ref;
    
    bgData = mean2(data(miny:maxy,minx:maxx));
    bgRef = mean2(ref(miny:maxy,minx:maxx));
    NormFactor = bgRef/bgData
    imagesc(data./ref*NormFactor);
    
    
    
    if abs(NormFactor-1)>0.1
        set(handles.textWarning,'String','The background intensity is off!')
    end
    
%     NormFactor = 1;
%     imagesc(data./ref);
    
    caxis([0.65 1.25])
end
colormap('gray')
axis equal
xlim(xlimit)
ylim(ylimit)
title(['Raw hologram of ',contents{get(hObject,'Value')}])

% Check if the measured hologram is saturated. 
if max(data(:)) >= 2^16
    set(handles.textWarning,'String','The measured hologram is saturated!')
else
    set(handles.textWarning,'String','The measured hologram is ok!')
end

handles.data = data;
handles.name = contents{get(hObject,'Value')};

guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in SetRef.
function SetRef_Callback(hObject, eventdata, handles)
% hObject    handle to SetRef (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = get(handles.listbox1,'String');
ref = double(imread([get(handles.text_dir,'String'),contents{get(handles.listbox1,'Value')}]));

set(handles.text_ref,'String',contents{get(handles.listbox1,'Value')});
handles.ref = ref;
guidata(hObject, handles);



% --- Executes on button press in Reconstruction.
function Reconstruction_Callback(hObject, eventdata, handles)
% hObject    handle to Reconstruction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% stop watch start
tstart = tic;
axes(handles.axes2)
title('Calculating ...')

pause(0.01)

Data = handles.data;
Ref = handles.ref;

% Setup info: Wavelength and sample-to-imager distance
lambda = str2double(get(handles.wavelength,'String'))*1e-9;    
Dz = str2double(get(handles.Dz,'String'))*1e-4;


delta2 = 2.2e-6;        % [m] pixel size of CCD
  
k = 2*pi/lambda;    % wavevector

% Get x and y-axis boundaries 
xlimit = get(handles.axes1,'Xlim');
ylimit = get(handles.axes1,'Ylim');

minx = ceil(xlimit(1));
maxx = floor(xlimit(2));
miny = ceil(ylimit(1));
maxy = floor(ylimit(2));

% Adjust min and max boundaries if the xlim and ylim is out of hologram image
if minx < 1 
    minx = 1;
end
if maxx > size(Data,2)
    maxx = size(Data,2);
end
if miny < 1
    miny = 1;
end
if maxy > size(Data,1)
    maxy = size(Data,1);
end


% Determine upsampling factor based on the image size
if maxy-miny <= 256
    UpsampleFactor = 2;
else if maxy-miny <= 512
    UpsampleFactor = 1;
else
    UpsampleFactor = 0;    
    end
end

% Force the upsampling factor to 2
UpsampleFactor = 2;

% When backround intensities are different between sample and ref images
 bgData = mean2(Data(miny:maxy,minx:maxx));
 bgRef = mean2(Ref(miny:maxy,minx:maxx));
 NormFactor = bgRef/bgData
% NormFactor = 1;

% Only reconstruct an image on the window to reduce computing time
% if the transmission method is selected, the input value (subNormAmp) for
% the iteration process should be normalized by the reference image. 

% subNormAmp = sqrt(Data(miny:maxy,minx:maxx)./(Ref(miny:maxy,minx:maxx)));
subNormAmp = sqrt(Data(miny:maxy,minx:maxx)./(Ref(miny:maxy,minx:maxx))*NormFactor);

        
delta_prev = delta2;

if UpsampleFactor > 0
    [subNormAmp,delta2] = upsampling(subNormAmp,UpsampleFactor,delta_prev);    
end

Nx = size(subNormAmp,1);
Ny = size(subNormAmp,2);
delta1 = delta2;

dfx = 1/(Nx*delta2);  % grid spacing in frequency domain
dfy = 1/(Ny*delta2);

[fx fy] = meshgrid((-Ny/2:Ny/2-1)*dfy,(-Nx/2:Nx/2-1)*dfx);


% Transformation function
Gbp = zeros(Nx,Ny);
Gfp = zeros(Nx,Ny);
for n = 1:size(fx,1)
    for m=1:size(fx,2)
        Gbp(n,m) = exp(1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
        Gfp(n,m) = exp(-1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
    end
end


% The number of iteration
NumIteration = str2double(get(handles.NumIter,'String'));      

% The measured hologram amplitude is used for an initial estimate
Input = subNormAmp;


%% Reconstruction process

for k=1:NumIteration
    
    % Fourier transform of function at the detector plane. 
    F2 = ft2(Input,delta2);      

    % Reconstructed image at the object plane
    Recon1 = ift2(F2.*Gbp,dfx,dfy);

    
%% Object support
    if k==1
       
        temp = zeros(size(Recon1));
        support = zeros(size(Recon1));
%         Threshold_objsupp = 0.06; 
%         Threshold_objsupp = 0.040; 

        Threshold_objsupp = str2double(get(handles.ThresholdObj,'String'));

        support = stdfilt(abs(Recon1).*cos(angle(Recon1)),ones(9));
        support = im2bw(support,Threshold_objsupp);
        se = strel('disk',6,0);
        support = imdilate(support,se);
        support = imfill(support,'holes');
        support = bwareaopen(support,300);
       
    end
   
%% Constraint
    
    % Preserve images inside the object support and set "1" to the
    % pixels outside of the object support
            
    Constraint = ones(size(Recon1));
    
    for p=1:size(Recon1,1)
        for q=1:size(Recon1,2)
            
            if support(p,q) == 1
                Constraint(p,q) = abs(Recon1(p,q));
            end
            
    % Transmission constraint
    % if the transmission is greater than 1, the value is set to unity. 
    % basic assumption is the normalized transmission (absorption) value
    % cannot be greater (lower) than 1 (0). If it happens, it is due to
    % inteference with its twin image. 

            if abs(Recon1(p,q))>1
                Constraint(p,q) = 1;                
            end
        end
    end
    

% togglePhase flipping 
% Zhao et al., Opt. Eng. 50, 091310 (2011)
% flip the togglephase of object by changing the sign of the exponential term
%     if k==1
%         Recon1_update = Constraint.*exp(-1i*angle(Recon1));
%     else
%         Recon1_update = Constraint.*exp(1i*angle(Recon1));
%     end
    
    Recon1_update = Constraint.*exp(1i*angle(Recon1));
    

    % Fourier transform of function at the object plane for transformation
    F1 = ft2(Recon1_update,delta1);
    
    % Reconstructed image at the detector plane
    Output = ift2(F1.*Gfp,dfx,dfy);

    % New input for the next iteration
    Input = subNormAmp.*exp(1i*angle(Output));
    
    
    % For the last iteration
    if k==NumIteration
        Output_Final = Input;
    end    
end

F2 = ft2(Output_Final,delta2);
ReconImage = ift2(F2.*Gbp,dfx,dfy);




Modulus = abs(ReconImage);
Phase = angle(ReconImage);

Phase_Revise = abs(Phase - mean2(Phase));
for p=1:size(ReconImage,1)
    for q=1:size(ReconImage,2)
        if Phase_Revise(p,q) > pi()
            Phase_Revise(p,q) = 2*pi()-Phase_Revise(p,q);
        end
    end
end

ExecTime = toc(tstart);
axes(handles.axes2)

if get(handles.radiobuttonColor,'Value')
    [m,n,r] = size(Modulus);
    ReconColor = zeros(m,n,3);
    ReconColor(:,:,2) = 1-abs(ReconImage);
    imagesc(ReconColor)
end


if get(handles.radiobuttonGray,'Value')
    imagesc(Modulus)
    colormap(gray)
end

axis equal        
caxis([0 1])
title(['Reconstructed amplitude after ',num2str(NumIteration),' iteration. Processing time = ', num2str(ExecTime),' sec'])




% update normalized hologram and reconstructed image shown on the screen
handles.Recon = ReconImage;
handles.Submap = subNormAmp;
handles.Factor = UpsampleFactor;
handles.Range = [minx maxx miny maxy];
handles.NormAmp = Data;
handles.stdmap = temp;
handles.ObjSupp = support;
handles.Phase = Phase_Revise;
% handles.Phase = Phase;

% Update handles structure
guidata(hObject, handles);




% --- Executes on button press in togglePhase.
function togglePhase_Callback(hObject, eventdata, handles)
% hObject    handle to togglePhase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglePhase


ReconImage = handles.Recon;
Phase = handles.Phase;

axes(handles.axes2)
xlimit = get(handles.axes2,'Xlim');
ylimit = get(handles.axes2,'Ylim');

minx = ceil(xlimit(1));
maxx = floor(xlimit(2));
miny = ceil(ylimit(1));
maxy = floor(ylimit(2));

% Adjust min and max boundaries if the xlim and ylim is out of hologram image
if minx < 1 
    minx = 1;
end
if maxx > size(ReconImage,2)
    maxx = size(ReconImage,2);
end
if miny < 1
    miny = 1;
end
if maxy > size(ReconImage,1)
    maxy = size(ReconImage,1);
end

Modulus = abs(ReconImage);

% Show the phage information when it is selected. Otherwise, show the
% amplitude (Modulus and togglephase)
TogglePhase = get(hObject,'Value');

if TogglePhase
    
    Phasebackground = mean2(Phase)*3;
%     Phasebackground = 0;
    if get(handles.radiobuttonColor,'Value')
    [m,n,r] = size(Modulus);
    ReconColor = zeros(m,n,3);
    ReconColor(:,:,1) = Phase-Phasebackground;
    imagesc(ReconColor)
    end


    if get(handles.radiobuttonGray,'Value')
    imagesc(Phase)
    colormap(hot)
    end

    axis equal
    title('Reconstructed phase')
    caxis([0 pi()])
   

else
    
    if get(handles.radiobuttonColor,'Value')
        [m,n,r] = size(Modulus);
        ReconColor = zeros(m,n,3);
        ReconColor(:,:,2) = 1-abs(ReconImage);
        imagesc(ReconColor)
    end


    if get(handles.radiobuttonGray,'Value')
        imagesc(Modulus)
        colormap(gray)
    end

    axis equal        
    caxis([0 1])
    title('Reconstructed amplitude')
    
end

xlim([minx maxx])
ylim([miny maxy])

guidata(hObject, handles);



% --- Executes on button press in toggleSupport.
function toggleSupport_Callback(hObject, eventdata, handles)
% hObject    handle to toggleSupport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of toggleSupport


ReconImage = handles.Recon;
support = handles.ObjSupp;
axes(handles.axes2)
xlimit = get(handles.axes2,'Xlim');
ylimit = get(handles.axes2,'Ylim');

minx = ceil(xlimit(1));
maxx = floor(xlimit(2));
miny = ceil(ylimit(1));
maxy = floor(ylimit(2));

% Adjust min and max boundaries if the xlim and ylim is out of hologram image
if minx < 1 
    minx = 1;
end
if maxx > size(ReconImage,2)
    maxx = size(ReconImage,2);
end
if miny < 1
    miny = 1;
end
if maxy > size(ReconImage,1)
    maxy = size(ReconImage,1);
end

% ToggleSupport = get(handles.Support,'Value');

if get(handles.toggleSupport,'Value')

    imagesc(support)
    axis equal
    title('Object support')
else
    imagesc(abs(ReconImage))
    axis equal
    title('Reconstructed amplitude')
    colormap('gray')
end

xlim([minx maxx])
ylim([miny maxy])







% --- Executes on button press in toggleComposite.
function toggleComposite_Callback(hObject, eventdata, handles)
% hObject    handle to toggleComposite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of toggleComposite

Phase = handles.Phase;

Phasebackground = mean2(Phase)*3;
Phase = Phase-Phasebackground;

ReconImage = handles.Recon;
Amp = 1-abs(ReconImage);

for i=1:size(Phase,1)
    for j=1:size(Phase,2)
        if Phase(i,j)<0
            Phase(i,j)=0;
        end
        
        if Amp(i,j)<0
            Amp(i,j)=0;
        end     
    end
end

Norm_Factor = 1;
    composite_image = imfuse(Norm_Factor*Amp,Phase,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);

if get(hObject,'Value')
    
    imagesc(composite_image)

else
     
    
        if get(handles.radiobuttonColor,'Value')
            [m,n,r] = size(ReconImage);
            ReconColor = zeros(m,n,3);
            ReconColor(:,:,2) = Amp;
            imagesc(ReconColor)
        end


        if get(handles.radiobuttonGray,'Value')
            imagesc(rgb2gray(composite_image))
            colormap(gray)
        end
    end
    axis equal        
%     caxis([0 1])
    title('Reconstructed amplitude')



% --- Executes on button press in toggleDetection.
function toggleDetection_Callback(hObject, eventdata, handles)
% hObject    handle to toggleDetection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tstart = tic;

axes(handles.axes2)
title('Detecting beads ...')
pause(0.01)



support = handles.ObjSupp;
Phase = handles.Phase;
Phasebackground = mean2(Phase)*3;
Phase = Phase-Phasebackground;



ReconImage = handles.Recon;
Amp = 1-abs(ReconImage);

for i=1:size(Phase,1)
    for j=1:size(Phase,2)
        if Phase(i,j)<0
            Phase(i,j)=0;
        end
        
        if Amp(i,j)<0
            Amp(i,j)=0;
        end     
    end
end

Norm_Factor = 2;
composite = imfuse(Norm_Factor*Amp,Phase,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);
I = rgb2gray(composite);

 

[pos_amp,radii_amp] = imfindcircles(Amp,[4,9],'ObjectPolarity','bright','Sensitivity',0.9);
[pos_phase,radii_phase] = imfindcircles(I,[10,30],'ObjectPolarity','bright','Sensitivity',0.85);
% [pos_amp,radii_amp] = imfindcircles(I,[10,30],'ObjectPolarity','bright','Sensitivity',0.85);

% [pos_phase,radii_phase] = imfindcircles(Phase,[7,24],'ObjectPolarity','bright','Sensitivity',0.9);


radii_phase


% [accum, posamp, radii_amp] = CircularHough_Grd(1-Amp,[3 9],0.01,3);
% [accum, pos_phase, radii_phase] = CircularHough_Grd(Phase,[3 9],0.005,10);
% [accum, pos_phase2, radii_phase2] = CircularHough_Grd(Phase,[10 20],0.005,10);



viscircles(pos_phase,radii_phase,'EdgeColor','w');

% round up the position coordinate
pos_amp = round(pos_amp);
pos_phase = round(pos_phase);

global IntCell PhaseCell
% global IntCell IntCell1 IntCell2 IntCell3 IntCell4 IntCell5 IntCell6 
% global PhaseCell1 PhaseCell2 PhaseCell3 PhaseCell4 PhaseCell5 PhaseCell6

IntCell = [];
PhaseCell = [];
% IntCell1 =[];
% IntCell2 = [];
% IntCell3 = [];
% IntCell4 = [];
% IntCell5 = [];
% IntCell6 = [];
% PhaseCell1 = [];
% PhaseCell2 = [];
% PhaseCell3 = [];
% PhaseCell4 = [];
% PhaseCell5 = [];
% PhaseCell6 = [];

mask = zeros(21,21);
for i=1:size(mask,1)
    for j=1:size(mask,2)
        if (i-11)^2+(j-11)^2 <= 49
            mask(i,j) = 1;
        end
    end
end

mask2 = zeros(31,31);
for i=1:size(mask,1)
    for j=1:size(mask2,2)
        if (i-15)^2+(j-15)^2 <= 15^2
            mask2(i,j) = 1;
        end
    end
end


%% Find beads

k_bead = 1;
margin = 30;
for i=1:size(pos_amp,1)
    if support(pos_amp(i,2),pos_amp(i,1))
        if radii_amp(i) < 10
            if pos_amp(i,1)>margin && pos_amp(i,2)>margin && pos_amp(i,1)+margin<size(Phase,2) && pos_amp(i,2)+margin<size(Phase,1)
                bead_phase = sum(sum(Phase(pos_amp(i,2)-5:pos_amp(i,2)+5,pos_amp(i,1)-5:pos_amp(i,1)+5)))/121;

                if bead_phase <= 0.5;

                    pos_bead(k_bead,:) = pos_amp(i,:);
%                     viscircles(pos_amp(i,:),radii_amp(i),'EdgeColor','b');
                    k_bead = k_bead+1;
                end
            end
        end
    end
end

            
    




%% Find Cells
k_cell = 1;
margin = 30;
for i=1:size(pos_phase,1)
    
    if support(pos_phase(i,2),pos_phase(i,1))
        if radii_phase(i)>=8
    
        if pos_phase(i,1)>margin && pos_phase(i,2)>margin && pos_phase(i,1)+margin<size(Phase,2) && pos_phase(i,2)+margin<size(Phase,1)
    
            IntCell(k_cell,1) = sum(sum(Amp(pos_phase(i,2)-10:pos_phase(i,2)+10,pos_phase(i,1)-10:pos_phase(i,1)+10).*mask))/sum(sum(mask));
            IntCell(k_cell,2) = sum(sum(Amp(pos_phase(i,2)-10:pos_phase(i,2)+10,pos_phase(i,1)-10:pos_phase(i,1)+10)));
            IntCell(k_cell,3) = sum(sum(Amp(pos_phase(i,2)-5:pos_phase(i,2)+5,pos_phase(i,1)-5:pos_phase(i,1)+5)));
            IntCell(k_cell,4) = sum(sum(Amp(pos_phase(i,2)-3:pos_phase(i,2)+3,pos_phase(i,1)-3:pos_phase(i,1)+3)));
            IntCell(k_cell,5) = Amp(pos_phase(i,2),pos_phase(i,1));
            IntCell(k_cell,6) = sum(sum(Amp(pos_phase(i,2)-15:pos_phase(i,2)+15,pos_phase(i,1)-15:pos_phase(i,1)+15)));
            IntCell(k_cell,7) = sum(sum(Amp(pos_phase(i,2)-15:pos_phase(i,2)+15,pos_phase(i,1)-15:pos_phase(i,1)+15).*mask2))/sum(sum(mask2));
            
            radii_phase(i) = round(radii_phase(i));
            mask3 = zeros(radii_phase(i)*2+1,radii_phase(i)*2+1);
            for a=1:size(mask3,1)
                for b=1:size(mask3,2)
                    if (a-radii_phase(i))^2+(b-radii_phase(i))^2 <= radii_phase(i)^2
                        mask3(a,b) = 1;
                    end
                end
            end
            mask3;
            IntCell(k_cell,8) = sum(sum(Amp(pos_phase(i,2)-radii_phase(i):pos_phase(i,2)+radii_phase(i),pos_phase(i,1)-radii_phase(i):pos_phase(i,1)+radii_phase(i)).*mask3))/sum(sum(mask3));
            
            mask4 = zeros(radii_phase(i)*2+1,radii_phase(i)*2+1);
            for a=1:size(mask4,1)
                for b=1:size(mask4,2)
                    if (a-radii_phase(i))^2+(b-radii_phase(i))^2 <= (radii_phase(i)-6)^2
                        mask4(a,b) = 1;
                    end
                end
            end
            IntCell(k_cell,9) = sum(sum(Amp(pos_phase(i,2)-radii_phase(i):pos_phase(i,2)+radii_phase(i),pos_phase(i,1)-radii_phase(i):pos_phase(i,1)+radii_phase(i)).*mask4))/sum(sum(mask4));

            PhaseCell(k_cell,1) = sum(sum(Phase(pos_phase(i,2)-10:pos_phase(i,2)+10,pos_phase(i,1)-10:pos_phase(i,1)+10).*mask))/sum(sum(mask));
            PhaseCell(k_cell,2) = sum(sum(Phase(pos_phase(i,2)-10:pos_phase(i,2)+10,pos_phase(i,1)-10:pos_phase(i,1)+10)));
            PhaseCell(k_cell,3) = sum(sum(Phase(pos_phase(i,2)-5:pos_phase(i,2)+5,pos_phase(i,1)-5:pos_phase(i,1)+5)));
            PhaseCell(k_cell,4) = sum(sum(Phase(pos_phase(i,2)-3:pos_phase(i,2)+3,pos_phase(i,1)-3:pos_phase(i,1)+3)));
            PhaseCell(k_cell,5) = Phase(pos_phase(i,2),pos_phase(i,1));
            PhaseCell(k_cell,6) = sum(sum(Phase(pos_phase(i,2)-15:pos_phase(i,2)+15,pos_phase(i,1)-15:pos_phase(i,1)+15)));
            
            
            pos_cell(k_cell,:) = pos_phase(i,:);
            rad_cell(k_cell) = radii_phase(i);

            viscircles(pos_phase(i,:),radii_phase(i),'EdgeColor','r');
            
            k_cell = k_cell+1;
        end
        end
    end
end

IntCell
sum(IntCell)/size(IntCell,1)

ExecTime = toc(tstart)
axes(handles.axes2)
title(['Beads/cells detected. Processing time = ', num2str(ExecTime),' sec'])



dist_threshold = 30; 
D=[];

Bead_cnt = zeros(size(pos_cell,1),1);


if k_cell>1
    if k_bead>1
        
        for i=1:size(pos_cell,1)
    

            D = pdist2(pos_cell(i,:),pos_bead,'euclidean');
   

            if ~isempty(D) 
                [rows,cols] = find(D<dist_threshold & D>rad_cell(i));
                
            else
                rows=[];
            end


            if ~isempty(rows)
                Bead_cnt(i) = size(rows,2);
                

            end
        end
    end
end

global pair

pair = [pos_cell(:,1),pos_cell(:,2),Bead_cnt];



c = clock;
directory_name = get(handles.text_dir,'String');
name = handles.name;
nameindex = strfind(name,'.png');

filename = [directory_name,'/',name(1:nameindex-1),'_',datestr(now,1),'_',num2str(c(4)),'-',num2str(c(5)),'.mat'];
save(filename,'directory_name','name','IntCell')




% handles.Beadcnt = Bead_cnt;

% Update handles structure
guidata(hObject, handles);



% --- Executes on button press in SaveAxis.
function SaveAxis_Callback(hObject, eventdata, handles)
% hObject    handle to SaveAxis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


name = handles.name;
index = get(handles.listbox1,'Value')
ImageAxis = handles.ImageAxis;
xlimit = get(handles.axes1,'Xlim');
ylimit = get(handles.axes1,'Ylim');

ImageAxis{index,1} = name;
ImageAxis{index,2} = xlimit;
ImageAxis{index,3} = ylimit;
ImageAxis

handles.ImageAxis = ImageAxis;
guidata(hObject, handles);






% --- Executes on button press in Auto.
function Auto_Callback(hObject, eventdata, handles)
% hObject    handle to Auto (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ImageAxis = handles.ImageAxis;
numFiles = size(ImageAxis,1);
ref = handles.ref;


for i=1:numFiles
    if size(ImageAxis{i,1},1)
        xlimit = ImageAxis{i,2};
        ylimit = ImageAxis{i,3};
        name = ImageAxis{i,1};
        data = double(imread([get(handles.text_dir,'String'),name]));
        axes(handles.axes1)
        imagesc(data./ref);
        colormap('gray')
        caxis([0.65 1.25])
        axis equal
        xlim(xlimit)
        ylim(ylimit)
        title(['Raw hologram of ',ImageAxis{i,1}])
        handles.data = data;
        handles.name = name;
        guidata(hObject, handles);
        
        Reconstruction_Callback(handles.Reconstruction, eventdata, handles)
        toggleDetection_Callback(handles.toggleDetection, eventdata, guidata(handles.Reconstruction))
        
    end
end




function ThresholdObj_Callback(hObject, eventdata, handles)
% hObject    handle to ThresholdObj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ThresholdObj as text
%        str2double(get(hObject,'String')) returns contents of ThresholdObj as a double


% --- Executes during object creation, after setting all properties.
function ThresholdObj_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ThresholdObj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in DataExport.
function DataExport_Callback(hObject, eventdata, handles)
% hObject    handle to DataExport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function ExportFile_Callback(hObject, eventdata, handles)
% hObject    handle to ExportFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ExportFile as text
%        str2double(get(hObject,'String')) returns contents of ExportFile as a double


% Export values selected in the list
check = zeros(8,1);
check(1) = get(handles.ExpDir,'Value'); 
check(2) = get(handles.ExpFile,'Value'); 
check(3) = get(handles.ExpHolo,'Value'); 
check(4) = get(handles.ExpRecon,'Value'); 
check(5) = get(handles.ExpRange,'Value'); 
check(6) = get(handles.ExpObj,'Value'); 
check(7) = get(handles.ExpAmp,'Value'); 
check(8) = get(handles.ExpComposite,'Value'); 


name=get(handles.ExportFile,'String');

ExportName = [name,'.mat'];



SaveFiles = matfile(ExportName,'Writable',true);

if check(1) == 1
    Directory = get(handles.text_dir,'String');
    SaveFiles.Directory = Directory;    
end
if check(2) == 1
    contents = get(handles.listbox1,'String');
    FileName = contents{get(handles.listbox1,'Value')};
    SaveFiles.FileName = FileName;    
end
if check(3) == 1
    subNormAmp = handles.Submap;
    SaveFiles.subNormAmp = subNormAmp;    
end
if check(4) == 1
    ReconImage = handles.Recon;
    SaveFiles.ReconImage = ReconImage;    
end
if check(5) == 1
    Range = handles.Range;
    SaveFiles.Range = Range;    
end
if check(6) == 1
    ObjSupp = handles.ObjSupp;
    SaveFiles.ObjSupp = ObjSupp;    
end
if check(7) == 1
    Directory = get(handles.text_dir,'String');
    Phase = handles.Phase; 
    ReconImage = handles.Recon;
    
    Norm_Factor = 1.5;
    

    [m,n,r] = size(ReconImage);
    AmpGreen = zeros(m,n,3);
    PhaseRed = zeros(m,n,3);
    PhaseRed(:,:,1) = Phase;
    AmpGreen(:,:,2) = Norm_Factor*(1-abs(ReconImage));
%     PhaseRed(1,1,1) = pi();
%     AmpGreen(1,1,2) = pi();
    imwrite(PhaseRed,[Directory name '_phase.tif']);
    imwrite(AmpGreen,[Directory name '_amp.tif']);
end

if check(8) == 1
    Directory = get(handles.text_dir,'String');
    Phase = handles.Phase; 
    ReconImage = handles.Recon;
    
    Norm_Factor = 1.5;
    composite = imfuse(Norm_Factor*(1-abs(ReconImage)),Phase,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);
    imwrite(composite,[Directory name '_composite.tif']);
end
        
% --- Executes during object creation, after setting all properties.
function ExportFile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ExportFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ExpDir.
function ExpDir_Callback(hObject, eventdata, handles)
% hObject    handle to ExpDir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpDir


% --- Executes on button press in ExpFile.
function ExpFile_Callback(hObject, eventdata, handles)
% hObject    handle to ExpFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpFile


% --- Executes on button press in ExpHolo.
function ExpHolo_Callback(hObject, eventdata, handles)
% hObject    handle to ExpHolo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpHolo


% --- Executes on button press in ExpRecon.
function ExpRecon_Callback(hObject, eventdata, handles)
% hObject    handle to ExpRecon (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpRecon


% --- Executes on button press in ExpRange.
function ExpRange_Callback(hObject, eventdata, handles)
% hObject    handle to ExpRange (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpRange


% --- Executes on button press in ExpObj.
function ExpObj_Callback(hObject, eventdata, handles)
% hObject    handle to ExpObj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpObj


% --- Executes on button press in ExpComposite.
function ExpComposite_Callback(hObject, eventdata, handles)
% hObject    handle to ExpComposite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpComposite


% --- Executes on button press in ExpAmp.
function ExpAmp_Callback(hObject, eventdata, handles)
% hObject    handle to ExpAmp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ExpAmp


function wavelength_Callback(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wavelength as text
%        str2double(get(hObject,'String')) returns contents of wavelength as a double


% --- Executes during object creation, after setting all properties.
function wavelength_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function NumIter_Callback(hObject, eventdata, handles)
% hObject    handle to NumIter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumIter as text
%        str2double(get(hObject,'String')) returns contents of NumIter as a double


% --- Executes during object creation, after setting all properties.
function NumIter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumIter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Dz_Callback(hObject, eventdata, handles)
% hObject    handle to Dz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Dz as text
%        str2double(get(hObject,'String')) returns contents of Dz as a double


% --- Executes during object creation, after setting all properties.
function Dz_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Dz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in DzScan.
function DzScan_Callback(hObject, eventdata, handles)
% hObject    handle to DzScan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.axes2)
title('Calculating ...')
pause(0.01)

Data = handles.data;
Ref = handles.ref;

lambda = str2double(get(handles.wavelength,'String'))*1e-9;    
Dz_int = str2double(get(handles.Dz,'String'))*1e-4;


delta2 = 2.2e-6;        % [m] pixel size of CCD
  
k = 2*pi/lambda;    % wavevector

% Get x and y-axis boundaries 
xlimit = get(handles.axes1,'Xlim');
ylimit = get(handles.axes1,'Ylim');

minx = ceil(xlimit(1));
maxx = floor(xlimit(2));
miny = ceil(ylimit(1));
maxy = floor(ylimit(2));

% Adjust min and max boundaries if the xlim and ylim is out of hologram image
if minx < 1 
    minx = 1;
end
if maxx > 2592
    maxx = 2592;
end
if miny < 1
    miny = 1;
end
if maxy > 1944
    maxy = 1944;
end


% Only reconstruct an image on the window to reduce computing time
% global subNormAmp
subNormAmp = sqrt(Data(miny:maxy,minx:maxx));
subRef = sqrt(Ref(miny:maxy,minx:maxx));

% if maxx-minx <= 256
%     UpsampleFactor = 2;
% else if maxx-minx <= 512
%     UpsampleFactor = 1;
% else
%     UpsampleFactor = 0;    
%     end
% end

UpsampleFactor = 2;
delta_prev = delta2;

if UpsampleFactor > 0
    [subNormAmp,delta2] = upsampling(subNormAmp,UpsampleFactor,delta_prev);
    [subRef,delta2] = upsampling(subRef,UpsampleFactor,delta_prev);
end

Nx = size(subNormAmp,1);
Ny = size(subNormAmp,2);
delta1 = delta2;

dfx = 1/(Nx*delta2);  % grid spacing in frequency domain
dfy = 1/(Ny*delta2);

[fx fy] = meshgrid((-Ny/2:Ny/2-1)*dfy,(-Nx/2:Nx/2-1)*dfx);


DzTemp = 0;

% global focus
% global Dzmin
Dzmin = Dz_int;
for Dzscan = 1:30
    Dz = Dz_int + Dzscan*0.1e-4

% Transformation function
Gbp = zeros(Nx,Ny);
Gfp = zeros(Nx,Ny);
for n = 1:size(fx,1)
    for m=1:size(fx,2)
        Gbp(n,m) = exp(1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
        Gfp(n,m) = exp(-1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
    end
end


% Step1. Define object support from transmission constraint
% Latychevskaia and Fink PRL 98, 233901 (2007)
% Latychevskaia and Fink Opt Exp 17, 10697 (2009)
% Zhao et al., Opt. Eng. 50, 091310 (2011)

% The number of iteration
% NumIteration = str2double(get(handles.NumIter,'String'));    
% NumIteration = 1;

% The measured hologram amplitude is used for an initial estimate
Input = subNormAmp./subRef;
% Input_Ref = subRef;

% Reconstruction process

    
    
    % Fourier transform of function at the detector plane. 
    F2 = ft2(Input,delta2);      
%     F2Ref = ft2(Input_Ref,delta2);
    % Reconstructed image at the object plane
    Recon1 = ift2(F2.*Gbp,dfx,dfy);
%     ReconRef = ift2(F2Ref.*Gbp,dfx,dfy);

        
    Modulus = abs(Recon1);
    temp_focus = Modulus;%Modulus(100:200,100:200);
%     temp_focus = Modulus(200:260,290:350);
    

    imagesc(Modulus);
    
    axis equal
    colormap(gray)
    title(['Dz = ',num2str(Dz)])
    caxis([0 1])
    pause(0.01)
    
        
        
    focus(Dzscan) = std(temp_focus(:));
    if focus(Dzscan) > DzTemp
        DzTemp = focus(Dzscan);
        Dzmin = Dz;
        Final = Modulus;
    
        imagesc(Modulus);
%         pause(0.1)
        axis equal
        caxis([0 1])
        colormap(gray)
        title(['Dz = ',num2str(Dzmin)])
        pause(0.01)

    end
end

imagesc(Final);
pause(0.1)
axis equal
caxis([0 1])
colormap(gray)
title(['Dz = ',num2str(Dzmin)])





function FileType_Callback(hObject, eventdata, handles)
% hObject    handle to FileType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FileType as text
%        str2double(get(hObject,'String')) returns contents of FileType as a double


% --- Executes during object creation, after setting all properties.
function FileType_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FileType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
































































%%%%%%%%%% HPV Tab %%%%%%

function Set_Callback(hObject, eventdata, handles)
% hObject    handle to Set (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Set as text
%        str2double(get(hObject,'String')) returns contents of Set as a double


% --- Executes during object creation, after setting all properties.
function Set_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Set (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Repeat_Callback(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Repeat as text
%        str2double(get(hObject,'String')) returns contents of Repeat as a double


% --- Executes during object creation, after setting all properties.
function Repeat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Start.
function Start_Callback(hObject, eventdata, handles)
% hObject    handle to Start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global circen cirrad mg_cnt ps_cnt pair_cnt Dist rows

% Choose a folder of D3 image
HPV_dir = uigetdir('//Home','Choose a folder to import...');
set(handles.text_Directory,'String',HPV_dir);
pause(0.01)

Set = str2double(get(handles.Set,'String'));
Repeat = str2double(get(handles.Repeat,'String'));

theFiles = dir([HPV_dir,'/*.png']);
numFiles = size(theFiles,1);
theData = cell(numFiles,1);
theNames = cell(numFiles,1);

% Check if the loaded image number is matched to set and repeat values
% if numFiles-1 == Set*Repeat
%     set(handles.text_NumFiles,'String',['Number of images = ',num2str(Set*Repeat)])
%     pause(0.01)
% else
%     set(handles.text_NumFiles,'String','ERROR!!! Mismatch in the file number')
%     return
% end


% read image data
for i = 1:numFiles
    indexfile = strfind(theFiles(i).name,'.png');
%     theData{i} = double(imread(fullfile(HPV_dir,theFiles(i).name)));
    theNames{i} = theFiles(i).name(1:indexfile-1);
    theData{i} = double(imread(fullfile(HPV_dir,theFiles(i).name)));
end

set(handles.listbox2,'String',theNames);

index = strfind(HPV_dir,'/D3');
% 
% ReconFolder = [HPV_dir(1:index(2)),'input/',HPV_dir(index(2)+4:size(HPV_dir,2))]
% ParameterFolder = [HPV_dir(1:index(2)),'input/HPV_Default']

% % % % copy image files to the GPU folder
% % % copyfile([HPV_dir,'/*.png'],ReconFolder,'f');
% % % 
% % % % copy reconstruction parameter files
% % % copyfile([ParameterFolder,'/*.*'],ReconFolder,'f');
% % % 
% % % set(handles.text_CopyFiles,'String','Copied files into the GPU folder')
% % % pause(0.01)
% % % 
% % % % wait for 10 min for image upload and reconstruction
% % % for i=1:10
% % %     set(handles.text_Hold,'String',['Wait for downloading - ',num2str(i-1),' of 10'])
% % %     pause(60)
% % % end
% % % set(handles.text_Hold,'String','Wait for 10 min for downloading - Done')

% check if all reconstructed images are downloaded

Result = zeros(Set*Repeat,5);
PairIndex = cell(Set*Repeat,1);
BeadCount = cell(Set*Repeat,1);
ReconNames = cell(Set*Repeat,1);


interval = 128;
kx = floor(size(theData{1},1)/interval)
ky = floor(size(theData{1},2)/interval)
ref = theData{numFiles};

pair_cnt = zeros(kx,ky);
ps_cnt = zeros(kx,ky);
mg_cnt = zeros(kx,ky);



for i=1:Set*Repeat
    
    
    
    for m = 1:kx
        for n = 1:ky
            
            
            
            
            rangex = (m-1)*interval+1:m*interval;
            rangey = (n-1)*interval+1:n*interval;
           
           
            
            
            ReconImage = Reconstruction(theData{i}(rangex,rangey),ref(rangex,rangey),405e-9,5e-4);
            axes(handles.axes3)
            
            [o,p,q] = size(ReconImage);
            ReconColor = zeros(o,p,3);
            ReconColor(:,:,2) = 1-abs(ReconImage);
            imagesc(ReconColor)
            
            
            axis equal        
            caxis([0 1])
            title([theNames{i}, '  ', num2str(rangex(1)), '-', num2str(rangey(1))])     
            pause(0.01)
            [accum, circen, cirrad] = CircularHough_Grd(1-abs(ReconImage),[3 8],0.01,3);
%             viscircles(circen,cirrad,'EdgeColor','b');

%             pause()
            
            circen = round(circen);
            Amp = abs(ReconImage);
            k_ps = 1;
            k_mg = 1;
            pos_ps=[];
            pos_mg=[];
            for k=1:size(circen,1)
%                 Amp(circen(k,2),circen(k,1))
                if Amp(circen(k,2),circen(k,1)) < 0.95
                    if cirrad(k) < 5.5
                        
                        pos_mg(k_mg,:) = circen(k,:);
                        k_mg = k_mg+1;
%                         viscircles(circen(k,:),cirrad(k),'EdgeColor','b');
                    else
                        pos_ps(k_ps,:) = circen(k,:);
                        k_ps = k_ps+1;
%                         viscircles(circen(k,:),cirrad(k),'EdgeColor','g');
                    end
                end
            end
           
            dist_threshold = 12; 
            
            Dist=[];
            

            if k_ps>1 && k_mg>1 
                Dist = pdist2(pos_mg,pos_ps,'euclidean');
            end
           
            

            if ~isempty(Dist) 
                [rows,cols] = find(Dist<dist_threshold);
                if size(rows,2)
                    pair_cnt(m,n) = size(rows,1);
                else
                    pair_cnt(m,n) = 0;
                end
            else
                pair_cnt(m,n) = 0;
            end
            ps_cnt(m,n) = k_ps-1;
            mg_cnt(m,n) = k_mg-1;
            
            title([theNames{i}, '  ', num2str(rangex(1)), '-', num2str(rangey(1)),' ps cnt: ', num2str(k_ps-1),'  mg cnt: ', num2str(k_mg-1),'  pair cnt: ', num2str(pair_cnt(m,n))]) 
            pause(0.01)
            
        end
    end
    BeadCount{i} = [ps_cnt, mg_cnt, pair_cnt];
    PS = sum(ps_cnt(:));
    MG = sum(mg_cnt(:));
    Pair = sum(pair_cnt(:));
    
    Result(i,1:4) = [PS MG Pair Pair/PS/MG*1e5];
    set(handles.uitable2,'Data',Result)
    pause(0.01)
end

t = handles.uitable2;
t.RowName = theNames(1:Set*Repeat);

row = get(handles.uitable2,'rowname');
col = get(handles.uitable2,'columnname')';
col = ['filename' col];

Result_Final = [row num2cell(Result)];
Result_Final = [col;Result_Final];

% export data
filename = [HPV_dir,'/Data_',HPV_dir(index(2)+4:size(HPV_dir,2)),'.mat']
save(filename,'HPV_dir','Result_Final','PairIndex','theNames')

set(handles.text_Export,'String','Exported data')    
    
    
%     theNames{i}    
%     ReconFiles = dir([ReconFolder,'/recon/',theNames{i},'*.png']);
%     numReconFiles = size(ReconFiles,1);
%     perImage = 315;
%     
%     BeadCount{i} = cell(perImage,3);
%     Names = cell(perImage,1);
%     
%     
% 
%     set(handles.text_inprocess,'String',['Processing - ', theNames{i}]);
%     
%     theData = cell(numReconFiles,1);
%         
%     for j=1:numReconFiles   
%         theData{j} = double(imread(fullfile([ReconFolder,'/recon/'],ReconFiles(j).name)));
%         nameindex = strfind(ReconFiles(j).name,'.png');
%         Names{j} = ReconFiles(j).name(1:nameindex(2)-1);
%     end
%     
%     ReconNames{i} = Names;
%     
%     
%     PS_cnt = zeros(numReconFiles,1);
%     Silica_cnt = zeros(numReconFiles,1);
%     num_pair = zeros(numReconFiles,1);
%     
%         
%     for k=1:numReconFiles
%         Data = theData{k};
%     
%         [PS_cnt(k),Silica_cnt(k),num_pair(k)] = HPV_Detection(Data,1,0);
%         if num_pair(k)
%             
%             % indicate the image number where any pair is found
%             PairIndex{i} = [PairIndex{i} k];
%             
%             axes(handles.axes3)
%             imagesc(abs(Data))
%             colormap(gray)
%             axis equal
%             title(['File number ',num2str(k),' - ',num2str(num_pair(k)),' pair/pairs'])
%             pause(0.01)
%             
%         end
%     end
%     
%     BeadCount{i} = [PS_cnt, Silica_cnt, num_pair];
%     PS = sum(PS_cnt(:));
%     Silica = sum(Silica_cnt(:));
%     Pair = sum(num_pair(:));
%     
%     Result(i,1:4) = [PS Silica Pair Pair/PS/Silica*1e5];
% %     set(handles.uitable2,'Data',Result)
% %     pause(0.01)
% 
%     if ~rem(i,Repeat)
%         SetNum = i/Repeat;
%         
%         Result((SetNum-1)*Repeat+1,5) = mean(Result((SetNum-1)*Repeat+1:SetNum*Repeat,4));
%     end      
%     set(handles.uitable2,'Data',Result)
%     pause(0.01)
%     
%     
%     
% end
% 
% set(handles.text_inprocess,'String','Processing - Done');
% 
% 
% 
% t = handles.uitable2;
% t.RowName = theNames(1:Set*Repeat);
% 
% row = get(handles.uitable2,'rowname');
% col = get(handles.uitable2,'columnname')';
% col = ['filename' col];
% 
% Result_Final = [row num2cell(Result)];
% Result_Final = [col;Result_Final];
% 
% % export data
% filename = [HPV_dir,'/Data_',HPV_dir(index+4:size(HPV_dir,2)),'.mat'];
% save(filename,'HPV_dir','Result_Final','PairIndex','BeadCount','ReconFolder','theNames','ReconNames')
% 
% set(handles.text_Export,'String','Exported data')
% 
% 
% % Calculation time
% ExecTime = toc(tstart);
% set(handles.text_CalTime,'String',['Calculation time = ',round(num2str(ExecTime)/60,1),' mins'])
% pause(0.01)
% 
% handles.ReconFolder = ReconFolder;
% handles.Result = Result_Final;
% handles.PairIndex = PairIndex;
% handles.BeadCount = BeadCount;
% handles.ReconNames = ReconNames;
% handles.theNames = theNames;
% guidata(hObject,handles)


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2

% PairIndex = handles.PairIndex;
% Result_Final = handles.Result;
% BeadCount = handles.BeadCount;
% ReconNames = handles.ReconNames;
% 
%   
% 
% % row_select = find(strcmp(Result_Final,ReconName));
% row_select = get(hObject,'Value');
% subPairIndex = PairIndex{row_select};
% subBeadCount = BeadCount{row_select};
% subReconNames = ReconNames{row_select};
% 
% set(handles.listbox3,'Value',size(subPairIndex,2));
% set(handles.listbox3,'String',subReconNames(subPairIndex));
% set(handles.uitable1,'Data',Result_Final(row_select+1,2:4));
% set(handles.uitable1,'rowname',Result_Final(row_select+1,1));
% 
% handles.subPairIndex = subPairIndex;
% handles.subBeadCount = subBeadCount;
% handles.subReconNames = subReconNames;
% guidata(hObject,handles);



% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox3.
function listbox3_Callback(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox3
% contents = cellstr(get(hObject,'String'));
% ReconName = contents{get(hObject,'Value')};
% 
% ReconFolder = handles.ReconFolder; 
% subPairIndex = handles.subPairIndex;
% subBeadCount = handles.subBeadCount;
% 
% index = strfind(ReconName,'.raw_');
% set(handles.uitable1,'rowname',ReconName(index+5:size(ReconName,2)));
% 
% index_selected = get(hObject,'Value');
% subPairIndex(index_selected);
% 
% Result = subBeadCount(subPairIndex(index_selected),:);
% set(handles.uitable1,'Data',Result);
% 
% 
% theData = double(imread(fullfile([ReconFolder,'/recon/'],[ReconName,'.png'])));        
% if theData
%     axes(handles.axes3)
%     imagesc(abs(theData))
%     colormap(gray)
%     axis equal
%     title(['File# ',num2str(subPairIndex(index_selected)),' - PS counts ',num2str(Result(1)),' - Silica counts ',num2str(Result(2)),' - Pair counts ',num2str(Result(3))])
% end


% --- Executes during object creation, after setting all properties.
function listbox3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in LoadData.
function LoadData_Callback(hObject, eventdata, handles)
% hObject    handle to LoadData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile('*.mat','Pick a mat file');
load([pathname, filename])


handles.Result = Result_Final;
handles.PairIndex = PairIndex;
handles.BeadCount = BeadCount;
handles.ReconFolder = ReconFolder;
handles.theNames = theNames;
handles.ReconNames = ReconNames;


set(handles.listbox1,'String',theNames)
set(handles.uitable2,'Data',Result_Final(2:size(Result_Final,1),2:size(Result_Final,2)))
set(handles.uitable2,'columnname',Result_Final(1,2:size(Result_Final,2)))
set(handles.uitable2,'rowname',Result_Final(2:size(Result_Final,1),1))



guidata(hObject,handles)
