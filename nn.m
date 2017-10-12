training_file = input('Enter the training file name: ', 's');
N = input('Enter the number of inputs (N): ');
M = input('Enter the number of outputs (M): ');
Nh = input('Enter the number of hidden units: ');

% The following code reads a text file and stores all the paterns in 
% an Nv by (N+M) matrix
fid = fopen(training_file, 'r');
training_file_values = fscanf(fid, '%f');
fclose(fid);
Nv = numel(training_file_values)/(N+M);
fprintf('Nv = %d\n', Nv);
training_file_values = reshape(training_file_values, [(N+M) Nv])';

% Store the inputs in variable x and the outputs in variable t
x = training_file_values(:, 1:N);
t = training_file_values(:, N+1:N+M);
clear training_file_values;


mx = zeros(N, 1);
sdx = zeros(N, 1);



%input mean and std vector

for p = 1:Nv
  for n = 1:N
    mx(n) = mx(n) + x(p,n);
    sdx(n) = sdx(n) + x(p,n)^2;
   end
  end

for i = 1:N
  mx(i) = mx(i)/Nv;
  sdx = sdx/(Nv-(mx(i))^2);
  sdx = sqrt(sdx);
end

%sub mx from input x
% N dimensional

sub_x = zeros(Nv,N);

for i =1:Nv
sub_x(i,:) = x(i) - mx';
end


file = input('Enter the file name: ', 's');

save("-mat",file,"sub_x") 

%randomize input weights

w = zeros(Nh, N+1);

m = 0;
randn ("seed", m);
for k = 1:Nh
  w(k,N+1) = randn(1,1);
  for n = 1:N
    w(k,n) = randn(1,1)/sdx(n);
  end
end



%net function

X = [ones(Nv,1) x];


nk = zeros(Nv,Nh);

for p = 1:Nv
  for k = 1:Nh
    for n = 1:N+1
  nk(p,k) = w(k,n)*X(p,n);
    end
  end
end

nk(1:5,:);

%mean and std of net fn.

mw = zeros(Nh, 1);
sdw = zeros(Nh, 1);

for p = 1:Nv
  for n = 1:N
    mw(n) = mw(n) + X(p,n);
    sdw(n) = sdw(n) + X(p,n)^2;
   end
  end

for i = 1:N
  mw(i) = mw(i)/Nv;
  sdw = sdw/(Nv-(mw(i))^2);
  sdw = sqrt(sdw);
end

%==========================

% perform net control

for p = 1:Nv
for k = 1:Nh
  
  nk(p,k) = nk(p,k)*(1/sdw(k));
  
end
end

% obtain mean and std of net control

nk(1:5,:)

mn = zeros(Nh, 1);
sdn = zeros(Nh, 1);

for p = 1:Nv
  for n = 1:Nh
    mn(n) = mn(n) + nk(p,n);
    sdn(n) = sdn(n) + nk(p,n)^2;
   end
  end

for i = 1:Nh
  mn(i) = mn(i)/Nv;
  sdn = sdn/(Nv-(mn(i))^2);
  sdn = sqrt(sdn);
end



mn
sdn