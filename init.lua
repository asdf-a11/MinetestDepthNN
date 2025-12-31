clearFileOnRun = true
writeToFile = true
writeFileString = minetest.get_worldpath().."/depth/out_new.txt"
playerHasJoined = false
savedDepthBufferCounter = 0
playerPressedKey = false
playerRef = nil
playerName = nil
timeBetween = 0.1
prevTime = 0

LoadFile = function(filePath, t)
	while true == true do
		local file, err = io.open(filePath, t)
		if file then
			return file
		end
		minetest.chat_send_all("file err ".. err)
	end
end

core.register_on_joinplayer(function(player)
	playerRef = player
	playerHasJoined = true
	playerName = player:get_player_name()
	core.chat_send_all("player Has Joined")
	if clearFileOnRun == true then
		local file = LoadFile(writeFileString, "w")
		io.close(file)
	end
end)

-- Vector utilities
function normalize(v)
	local mag = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
	return { x = v.x/mag, y = v.y/mag, z = v.z/mag }
end

function cross(a, b)
	return {
			x = a.y * b.z - a.z * b.y,
			y = a.z * b.x - a.x * b.z,
			z = a.x * b.y - a.y * b.x
	}
end

function subtract(a, b)
	return { x = a.x - b.x, y = a.y - b.y, z = a.z - b.z }
end

function add(a, b)
	return { x = a.x + b.x, y = a.y + b.y, z = a.z + b.z }
end

function scale(v, s)
	return { x = v.x * s, y = v.y * s, z = v.z * s }
end

-- Ray direction generator
function generateRayDirection(x, y, width, height, fov, cameraDir, cameraUp)
	local aspectRatio = width / height
	local halfHeight = math.tan(fov / 2)
	local halfWidth = aspectRatio * halfHeight

	local w = normalize(cameraDir)
	local u = normalize(cross(cameraUp, w))  -- right vector
	local v = cross(w, u)  -- corrected up vector

	-- NDC coordinates (-1 to 1)
	local ndcX = (2 * ((x + 0.5) / width) - 1) * halfWidth
	local ndcY = (1 - 2 * ((y + 0.5) / height)) * halfHeight

	local rayDir = normalize(add(add(scale(u, ndcX), scale(v, ndcY)), w))
	return rayDir
end
function deg2rad(deg)
	return deg * math.pi / 180
end

function getCameraDirection(yaw, pitch)
	--local yaw = deg2rad(yawDeg)
	--local pitch = deg2rad(pitchDeg)

	local dir = {
			x = math.cos(pitch) * math.sin(yaw),
			y = math.sin(pitch),
			z = -math.cos(pitch) * math.cos(yaw)
	}
	return normalize(dir)
end

worldUp = {x = 0, y = 1, z = 0}
function getCameraBasis(yawDeg, pitchDeg)
	local cameraDir = getCameraDirection(yawDeg, pitchDeg)
	local right = normalize(cross(worldUp, cameraDir))
	local cameraUp = cross(cameraDir, right)
	return cameraDir, cameraUp
end
function computeCameraUp(cameraDir)
	local worldUp = {x = 0, y = 1, z = 0}

	-- handle edge case: cameraDir parallel to worldUp
	local dot = cameraDir.x * worldUp.x + cameraDir.y * worldUp.y + cameraDir.z * worldUp.z
	if math.abs(dot) > 0.999 then
			worldUp = {x = 0, y = 0, z = 1} -- pick another up to avoid zero cross
	end

	local right = normalize(cross(worldUp, cameraDir))
	local cameraUp = cross(cameraDir, right)
	return normalize(cameraUp)
end
function GetMag(ray)
	return ray.x*ray.x + ray.y*ray.y + ray.z*ray.z
end

--minetest.register_globalstep(function(dtime) end)
function dump(o)
	if type(o) == 'table' then
		 local s = '{ '
		 for k,v in pairs(o) do
				if type(k) ~= 'number' then k = '"'..k..'"' end
				s = s .. '['..k..'] = ' .. dump(v) .. ','
		 end
		 return s .. '} '
	else
		 return tostring(o)
	end
end

local function work()
  minetest.after(timeBetween, work)
	if playerHasJoined == true then
		local playerRef = minetest.get_player_by_name("singleplayer")
		if playerRef:get_player_control().sneak == true then
			playerPressedKey = true
		end
		if playerPressedKey == true and os.time() - prevTime >= 1 then
			local file
			if writeToFile == true then
				file = LoadFile(writeFileString, "a")
				io.output(file)	
			end
			
			savedDepthBufferCounter = savedDepthBufferCounter + 1
			--minetest.chat_send_all("saved number = " .. tostring(savedDepthBufferCounter))	
			local player = minetest.get_player_by_name("singleplayer")
			local currentPos = player:getpos()
			local camUnitVector = player:get_look_dir()
			currentPos.y = currentPos.y + 1.75  -- + camUnitVector.y
			currentPos.x = currentPos.x + 0--  + camUnitVector.x
			currentPos.z = currentPos.z + 0--  + camUnitVector.z 0.5
			local fov = math.rad(110.0)	
			local screenRes = math.pow(2,6)
			local screenx = screenRes
			local screeny = screenRes
			local maxDepth = 100

			--minetest.chat_send_all(tostring(player:get_look_vertical() / (2 * 3.1415) * 360) .. " " .. tostring(player:get_look_horizontal() / (2 * 3.1415) * 360))
			minetest.chat_send_all(tostring(savedDepthBufferCounter))
			prevTime = os.time()

			local cameraDir = player:get_look_dir()
			local cameraUp = computeCameraUp(cameraDir)

			for y = 0, screeny-1, 1 do
				for x = 0, screenx-1, 1 do				
					--local xTheta = (y / screeny - 0.5) * fov
					--local yTheta = (x / screenx - 0.5) * fov
	
					--xTheta = xTheta - player:get_look_vertical()
					--yTheta = yTheta - player:get_look_horizontal()

					local ray = generateRayDirection(x, y, screenx, screeny, fov, cameraDir, cameraUp)

					ray = scale(ray, maxDepth)
					
					local newPos = {
						x = ray.x + currentPos.x,
						y = ray.y + currentPos.y,
						z = ray.z + currentPos.z
					}
					--local out, pos = minetest.line_of_sight(currentPos, newPos)
					local hit = true
					local pos = nil
					local ray = minetest.raycast(currentPos, newPos, false, true)
					if ray == nil then
						hit = false;
					else
						local pointable = ray:next()
						--minetest.chat_send_all(dump(pointable))
						if pointable == nil then
							hit = false
						else
							pos = pointable.intersection_point
						end						
					end
					local dist = maxDepth
					if hit == true then
						--minetest.set_node(pos, { name = "mcl_nether:glowstone" })
						local distX = (pos.x - currentPos.x) * (pos.x - currentPos.x)
						local distY = (pos.y - currentPos.y) * (pos.y - currentPos.y)
						local distZ = (pos.z - currentPos.z) * (pos.z - currentPos.z)
						dist = math.min(maxDepth, math.sqrt(distX + distY + distZ))	
					end
					if writeToFile == true then
						--file.write(tostring(pos))
						io.write(tostring(dist) .. "| ")
					end
	
					--minetest.chat_send_all(tostring(out) .. ", " .. tostring(pos))
				end	
			end	
			if writeToFile == true then
				io.write("? ")
				file:close()
			end
		end		
	end
end
minetest.after(2, work)